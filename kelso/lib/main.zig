const std = @import("std");
const py = @import("python.zig");

const Np = @import("numpy_data.zig");

var np: Np = undefined;

var _arena: std.heap.ArenaAllocator = undefined;
var global_arena: std.mem.Allocator = undefined;

var generator_error: ?*py.PyObject = null;

// This python module will export three functions
const module_methods = [_]py.PyMethodDef{
    .{
        .ml_name = "set_ontology",
        .ml_meth = set_ontology,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "",
    },
    .{
        .ml_name = "find_neighbours",
        .ml_meth = _find_neighbours,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "inputs a patient (ids + count), a list of patients (list of ids + list of counts) and a neighborhood size",
    },
    .{
        .ml_name = "test",
        .ml_meth = my_test,
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

export fn my_test(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    _ = self_obj;
    var arg: ?*py.PyObject = undefined;

    if (py.PyArg_ParseTuple(args, "O", &arg) == 0) return null;

    if (py.PyList_Check(arg) == 0) {
        py.PyErr_SetString(generator_error, "we were expecting a list...");
        return null;
    }
    const list: *py.PyListObject = @ptrCast(arg orelse return null);
    const size: usize = @intCast(py.PyList_GET_SIZE(@ptrCast(list)));

    for (0..size) |it| {
        const item = py.PyList_GetItem(@ptrCast(list), @intCast(it));
        const val = py.PyLong_AsLong(item);
        std.debug.print("item {}: {}\n", .{ it, val });
    }

    std.debug.print("list size: {}\n", .{size});

    py.Py_INCREF(py.Py_None);
    return py.Py_None;
}

// useful link: https://stackoverflow.com/questions/50668981/how-to-return-a-list-of-ints-in-python-c-api-extension-with-pylist

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

var ontology: []u32 = undefined;

/// inputs a patient (ids + count), a list of patients (list of ids + list of counts) and a neighborhood size
export fn _find_neighbours(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    _ = self_obj;
    var arg1: ?*py.PyObject = undefined;
    var arg2: ?*py.PyObject = undefined;
    var arg3: ?*py.PyObject = undefined;
    var arg4: ?*py.PyObject = undefined;
    var _num_neigh: i64 = undefined;

    if (py.PyArg_ParseTuple(args, "OOOOl", &arg1, &arg2, &arg3, &arg4, &_num_neigh) == 0) return null;

    const patient_codes = arg1;
    const patient_count = arg2;
    const dataset_codes = arg3;
    const dataset_count = arg4;
    const num_neigh: usize = @intCast(_num_neigh);
    _ = num_neigh;

    if (py.PyList_Check(arg3) == 0) {
        py.PyErr_SetString(generator_error, "Argument 3 should be a list");
        return null;
    }
    if (py.PyList_Check(arg4) == 0) {
        py.PyErr_SetString(generator_error, "Argument 4 should be a list");
        return null;
    }

    const list_len = blk: {
        const _list_len = py.PyList_GET_SIZE(dataset_codes);
        if (_list_len != py.PyList_GET_SIZE(dataset_count)) {
            py.PyErr_SetString(generator_error, "Ids list and count list should have the same size");
            return null;
        }
        break :blk @as(usize, @intCast(_list_len));
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const patient = parse_patient(patient_codes, patient_count, allocator) orelse {
        py.PyErr_SetString(generator_error, "Error while parsing patient");
        return null;
    };

    const dataset = blk: {
        var dataset = allocator.alloc([][]u32, list_len) catch return null;
        for (0..list_len) |it| {
            const codes_obj = py.PyList_GetItem(dataset_codes, @intCast(it));
            const count_obj = py.PyList_GetItem(dataset_count, @intCast(it));
            dataset[it] = parse_patient(codes_obj, count_obj, allocator) orelse {
                const msg = std.fmt.allocPrintZ(global_arena, "Error while parsing dataset patient {}", .{it}) catch return null;
                py.PyErr_SetString(generator_error, msg);
                return null;
            };
        }
        break :blk dataset;
    };

    find_neighbours(patient, dataset, ontology);

    py.Py_INCREF(py.Py_None);
    return py.Py_None;
}

export fn set_ontology(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    _ = self_obj;
    var arg1: ?*py.PyObject = undefined;

    if (py.PyArg_ParseTuple(args, "O", &arg1) == 0) return null;
    const _arr = np.from_otf(arg1, Np.Types.UINT, Np.Array_Flags.IN_ARRAY) orelse return null;
    const arr: *Np.Array_Obj = @ptrCast(_arr);

    var size: usize = undefined;
    var tree: []u32 = undefined;
    {
        const data: [*]u32 = @ptrCast(@alignCast(arr.data));
        const nd = arr.nd;
        if (nd != 2) {
            py.PyErr_SetString(generator_error, "Array should have a single dimension");
            return null;
        }
        const num_fields: usize = @intCast(arr.dimensions[1]);
        if (num_fields != 2) {
            const message = std.fmt.allocPrint(
                global_arena,
                "Array should have dim (*, 2), found dim ({}, {})",
                .{ arr.dimensions[0], arr.dimensions[1] },
            ) catch "Error";
            py.PyErr_SetString(generator_error, @ptrCast(message));
            return null;
        }
        size = @intCast(arr.dimensions[0]);
        tree = data[0 .. 2 * size];
    }

    const max_id = blk: {
        var max: u32 = 0;
        for (0..size) |it| {
            max = @max(tree[2 * it], max);
        }
        break :blk max;
    };

    ontology = std.heap.page_allocator.alloc(u32, max_id + 1) catch return null;
    for (0..size) |it| {
        const child = tree[2 * it];
        const parent = tree[2 * it + 1];
        ontology[child] = parent;
    }

    py.Py_INCREF(py.Py_None);
    return py.Py_None;
}

fn parse_patient(codes: ?*py.PyObject, counts: ?*py.PyObject, allocator: std.mem.Allocator) ?[][]u32 {
    const patient_id = blk: {
        const _patient_id = np.from_otf(codes, Np.Types.UINT, Np.Array_Flags.IN_ARRAY);
        break :blk @as(*Np.Array_Obj, @ptrCast(_patient_id orelse return null));
    };
    const patient_cc = blk: {
        const _patient_cc = np.from_otf(counts, Np.Types.UINT, Np.Array_Flags.IN_ARRAY);
        break :blk @as(*Np.Array_Obj, @ptrCast(_patient_cc orelse return null));
    };
    if (patient_id.nd != 1) return null;
    if (patient_cc.nd != 1) return null;
    const num_visits: usize = @intCast(patient_cc.dimensions[0]);
    var patient = allocator.alloc([]u32, num_visits) catch return null;

    const visit_lens: []u32 = @as([*]u32, @ptrCast(@alignCast(patient_cc.data)))[0..num_visits];
    const data: [*]u32 = @ptrCast(@alignCast(patient_id.data));

    var cursor: usize = 0;
    for (visit_lens, 0..) |c, it| {
        const len: usize = @intCast(c);
        patient[it] = data[cursor .. cursor + len];
        cursor += len;
    }

    return patient;
}

// this is chosen looking at the input ontology
const genealogy_max_size: usize = 12;

fn compute_c2c(id_1: u32, id_2: u32, _ontology: []u32) f32 {
    const res_1 = get_genealogy(id_1, _ontology);
    const res_2 = get_genealogy(id_2, _ontology);

    const genealogy_1 = res_1[0];
    const genealogy_2 = res_2[0];
    const root_1 = res_1[1];
    const root_2 = res_2[1];

    var cursor_1 = root_1;
    var cursor_2 = root_2;
    while (genealogy_1[cursor_1] == genealogy_2[cursor_2]) {
        cursor_1 -= 1;
        cursor_2 -= 1;
    }
    cursor_1 = @min(cursor_1 + 1, root_1);
    cursor_2 = @min(cursor_2 + 1, root_2);

    const d_lr_doubled: f32 = @floatFromInt(2 * (root_1 - cursor_1));
    const dist = d_lr_doubled / (@as(f32, @floatFromInt(cursor_1 + cursor_2)) + d_lr_doubled);

    return dist;
}

fn get_genealogy(id: u32, _ontology: []u32) struct { [genealogy_max_size]u32, usize } {
    var res = std.mem.zeroes([genealogy_max_size]u32);
    res[0] = id;
    var it: usize = 0;
    while (true) {
        const parent = _ontology[res[it]];
        if (parent != res[it]) {
            it += 1;
            res[it] = parent;
        } else break;
    }
    return .{ res, it };
}

fn asymmetrical_v2v(v1: []u32, v2: []u32, _ontology: []u32) f32 {
    var sum: f32 = 0;
    for (v1) |c1| {
        var best = std.math.floatMax(f32);
        for (v2) |c2| {
            const dist = compute_c2c(c1, c2, _ontology);
            best = @min(best, dist);
        }
        sum += best;
    }
    return sum;
}

fn compute_v2v(v1: []u32, v2: []u32, _ontology: []u32) f32 {
    const x = asymmetrical_v2v(v1, v2, _ontology);
    const y = asymmetrical_v2v(v2, v1, _ontology);
    return 0.5 * (x + y);
}

fn compute_p2p(p1: [][]u32, p2: [][]u32, _ontology: []u32) f32 {
    var table = std.heap.page_allocator.alloc(f32, p1.len * p2.len) catch @panic("error with allocation");
    defer std.heap.page_allocator.free(table);

    const w = p1.len;

    for (0..p1.len) |it| {
        for (0..p2.len) |jt| {
            table[jt * w + it] = std.math.floatMax(f32);
        }
    }

    for (0..p1.len) |it| {
        for (0..p2.len) |jt| {
            const cost = compute_v2v(p1[it], p2[jt], _ontology);
            const in_cost = if (it > 0) {
                break table[jt * w + it - 1];
            } else {
                break 0;
            };
            const del_cost = if (jt > 0) {
                break table[(jt - 1) * w + it];
            } else {
                break 0;
            };
            const edit_cost = if (it > 0 and jt > 0) {
                break table[(jt - 1) * w + it - 1];
            } else {
                break 0;
            };
            table[jt * w + it] = cost + @min(in_cost, del_cost, edit_cost);
        }
    }
    return table[table.len - 1];
}

fn find_neighbours(patient: [][]u32, dataset: [][][]u32, _ontology: []u32) void {
    _ = _ontology;
    _ = dataset;
    _ = patient;
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

    generator_error = py.PyErr_NewException("generator.error", null, null);
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

    _arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    global_arena = _arena.allocator();

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
