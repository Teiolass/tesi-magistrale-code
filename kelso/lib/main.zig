const std = @import("std");
const py = @import("python.zig");

const Np = @import("numpy_data.zig");

var np: Np = undefined;

var arena: std.heap.ArenaAllocator = undefined;
var global_arena: std.mem.Allocator = undefined;

var generator_error: ?*py.PyObject = null;

// This python module will export three functions
const module_methods = [_]py.PyMethodDef{
    .{
        .ml_name = "get_parent",
        .ml_meth = get_genealogy,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "",
    },
    .{ // this one is just a sentinel
        .ml_name = null,
        .ml_meth = null,
        .ml_flags = 0,
        .ml_doc = null,
    },
};

// export fn spam_system(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
//     _ = self_obj;
//     var command: [*c]u8 = undefined;
//     if (py.PyArg_ParseTuple(args, "s", &command) == 0) return null;
//     const sts = py.system(command);
//     if (sts < 0) {
//         py.PyErr_SetString(spam_error, "System command failed!");
//         return null;
//     }
//     return py.PyLong_FromLong(sts);
// }

// export fn np_ex(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
//     _ = self_obj;
//     var arg1: ?*py.PyObject = undefined;

//     if (py.PyArg_ParseTuple(args, "O", &arg1) == 0) return null;
//     const _arr = np.from_otf(arg1, Np.Types.LONG, Np.Array_Flags.INOUT_ARRAY2) orelse return null;
//     const arr: *Np.Array_Obj = @ptrCast(_arr);
//     std.debug.print("num_dimensions: {}\n", .{arr.nd});

//     const data: [*]i64 = @ptrCast(@alignCast(arr.data));
//     data[0] = 9;

//     py.Py_INCREF(py.Py_None);
//     return py.Py_None;
// }

// export fn get_nines(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
//     _ = self_obj;
//     const second_dimension_size = 4;
//     var arr_size: i64 = undefined;
//     if (py.PyArg_ParseTuple(args, "l", &arr_size) == 0) return null;

//     var dims = [_]isize{ arr_size, second_dimension_size };
//     var obj = np.simple_new(2, &dims, Np.Types.FLOAT) orelse return null;
//     {
//         var arr: *Np.Array_Obj = @ptrCast(obj);
//         var data: [*]f32 = @ptrCast(@alignCast(arr.data));
//         const num_els: usize = @intCast(arr_size * second_dimension_size);
//         for (0..num_els) |it| {
//             data[it] = 9.9;
//         }
//     }
//     return obj;
// }

export fn get_genealogy(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    _ = self_obj;
    var arg1: ?*py.PyObject = undefined;
    var id: i64 = undefined;

    if (py.PyArg_ParseTuple(args, "Ol", &arg1, &id) == 0) return null;
    const _arr = np.from_otf(arg1, Np.Types.UINT, Np.Array_Flags.IN_ARRAY) orelse return null;
    const arr: *Np.Array_Obj = @ptrCast(_arr);

    const tree: []u32 = blk: {
        const data: [*]u32 = @ptrCast(@alignCast(arr.data));
        const nd = arr.nd;
        if (nd != 1) {
            py.PyErr_SetString(generator_error, "Array should have a single dimension");
            return null;
        }
        const size: usize = @intCast(arr.dimensions[0]);
        break :blk data[0..size];
    };

    std.debug.print("Length of ontology is {}. Id is {}\n", .{ tree.len, id });

    const parent = tree[@intCast(id)];
    std.debug.print("parent id: {}\n", .{parent});

    py.Py_INCREF(py.Py_None);
    return py.Py_None;
}

var generator_module = py.PyModuleDef{
    .m_base = .{
        .ob_base = .{ .ob_refcnt = 1, .ob_type = null },
        .m_init = null,
        .m_index = 0,
        .m_copy = null,
    },
    .m_name = "generator",
    .m_doc = "",
    .m_size = -1,
    .m_methods = @constCast(&module_methods),
    .m_slots = null,
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

pub export fn PyInit_generator() ?*py.PyObject {
    const m = py.PyModule_Create(@constCast(&generator_module));
    if (m == null) return null;

    generator_error = py.PyErr_NewException("spam.error", null, null);
    py.Py_XINCREF(generator_error);
    if (py.PyModule_AddObject(m, "error", generator_error) < 0) {
        py.Py_XDECREF(generator_error);
        {
            const tmp = generator_error;
            if (tmp != null) {
                generator_error = null;
                py.Py_DECREF(tmp);
            }
        }
        py.Py_DECREF(m);
        return null;
    }

    np = import_numpy() catch std.debug.panic("cannot import numpy", .{});

    arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    return m;
}

fn import_numpy() !Np {
    // @todo this can fail in so many ways... See `__multiarray_api.h` line 1483
    const numpy = py.PyImport_ImportModule("numpy.core._multiarray_umath");
    if (numpy == null) return error.generic;
    const c_api = py.PyObject_GetAttrString(numpy, "_ARRAY_API");
    if (c_api == null) return error.generic;
    const PyArray_api = blk: {
        const t = py.PyCapsule_GetPointer(c_api, null);
        if (t == null) return error.generic;
        const ret: [*]?*void = @ptrCast(@alignCast(t));
        break :blk ret;
    };
    return Np.from_api(PyArray_api);
}
