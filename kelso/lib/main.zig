// this is chosen looking at the input ontology
const genealogy_max_size: usize = 12;
const num_jobs: usize = 1;

var np: Np = undefined;

var _arena: std.heap.ArenaAllocator = undefined;
var global_arena: std.mem.Allocator = undefined;

var generator_error: ?*py.PyObject = null;

// This python module will export three functions
const module_methods = [_]py.PyMethodDef{
    .{
        .ml_name = "create_c2c_table",
        .ml_meth = create_c2c_table,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "",
    },
    .{
        .ml_name = "find_neighbours",
        .ml_meth = _find_neighbours,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "inputs a patient (ids + count), a list of patients (list of ids + list of counts) and a neighborhood size",
    },
    .{ // this one is just a sentinel
        .ml_name = null,
        .ml_meth = null,
        .ml_flags = 0,
        .ml_doc = null,
    },
};

// useful link for c apis: https://stackoverflow.com/questions/50668981/how-to-return-a-list-of-ints-in-python-c-api-extension-with-pylist

var ontology: Ontology = undefined;

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

    var result_data: []f32 = undefined;
    const result_array = blk: {
        var dimensions = [_]isize{@intCast(dataset.len)};
        var obj = np.simple_new(dimensions.len, &dimensions, Np.Types.FLOAT) orelse {
            py.PyErr_SetString(generator_error, "Failed while creating result array");
            return null;
        };
        var arr: *Np.Array_Obj = @ptrCast(obj);
        result_data = @as([*]f32, @ptrCast(@alignCast(arr.data)))[0..dataset.len];
        break :blk obj;
    };

    find_neighbours(patient, dataset, ontology, result_data);

    // @debug
    // py.Py_INCREF(py.Py_None);
    // return py.Py_None;
    return result_array;
}

const Ontology = struct {
    table: []f32,
    w: usize,

    inline fn get(self: Ontology, c1: u32, c2: u32) f32 {
        return self.table[c1 * self.w + c2];
    }

    inline fn get_mut(self: Ontology, c1: u32, c2: u32) *f32 {
        return &self.table[c1 * self.w + c2];
    }
};

export fn create_c2c_table(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    _ = self_obj;
    var arg1: ?*py.PyObject = undefined;
    var arg2: ?*py.PyObject = undefined;

    if (py.PyArg_ParseTuple(args, "OO", &arg1, &arg2) == 0) return null;
    const _arr = np.from_otf(arg1, Np.Types.UINT, Np.Array_Flags.IN_ARRAY) orelse return null;
    const arr: *Np.Array_Obj = @ptrCast(_arr);

    const _arr2 = np.from_otf(arg2, Np.Types.UINT, Np.Array_Flags.IN_ARRAY) orelse return null;
    const arr2: *Np.Array_Obj = @ptrCast(_arr2);

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

    const leaf_indices: []u32 = blk: {
        const t: [*]u32 = @ptrCast(@alignCast(arr2.data));
        const _size: usize = @intCast(arr2.dimensions[0]); // @todo check nd
        break :blk t[0.._size];
    };
    const max_leaf_index = std.mem.max(u32, leaf_indices) + 1;

    const max_id = blk: {
        var max: u32 = 0;
        for (0..size) |it| {
            max = @max(tree[2 * it], max);
        }
        break :blk max;
    };

    var _ontology_tree = std.heap.page_allocator.alloc(u32, max_id + 1) catch return null;
    defer std.heap.page_allocator.free(_ontology_tree);
    for (0..size) |it| {
        const child = tree[2 * it];
        const parent = tree[2 * it + 1];
        _ontology_tree[child] = parent;
    }

    ontology = std.mem.zeroes(Ontology);
    ontology.table = global_arena.alloc(f32, max_leaf_index * max_leaf_index) catch @panic("Alloc error");
    ontology.w = max_leaf_index;
    for (ontology.table) |*it| {
        it.* = std.math.nan(f32);
    }

    if (comptime num_jobs == 1) {
        for (leaf_indices, 0..) |it, it_index| {
            for (leaf_indices[0 .. it_index + 1]) |jt| {
                const dist = compute_c2c(it, jt, _ontology_tree);
                ontology.get_mut(it, jt).* = dist;
                ontology.get_mut(jt, it).* = dist;
            }
        }
    } else {
        const Thread_Data = struct {
            ontology_tree: []u32,
            leaf_indices: []u32,
            ontology: Ontology,
            start: usize,
            len: usize,

            const Self = @This();

            pub fn job(self: Self) void {
                std.debug.print("hey im working!\n", .{});
                for (self.leaf_indices[self.start .. self.start + self.len], 0..) |it, it_index| {
                    for (self.leaf_indices[0 .. self.start + it_index + 1]) |jt| {
                        const dist = compute_c2c(it, jt, self.ontology_tree);
                        ontology.get_mut(it, jt).* = dist;
                        ontology.get_mut(jt, it).* = dist;
                    }
                }
            }
        };

        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        const spawn_config = std.Thread.SpawnConfig{ .allocator = allocator };

        const base_size = leaf_indices.len / num_jobs;
        var remainder = leaf_indices.len % num_jobs;

        var threads = allocator.alloc(std.Thread, num_jobs) catch @panic("error in allocation");

        var cursor: usize = 0;
        for (threads) |*thread| {
            var true_size = base_size;
            if (remainder > 0) {
                true_size += 1;
                remainder -= 1;
            }
            const thread_data = Thread_Data{
                .ontology_tree = _ontology_tree,
                .leaf_indices = leaf_indices,
                .ontology = ontology,
                .start = cursor,
                .len = true_size,
            };
            cursor += true_size;
            thread.* = std.Thread.spawn(spawn_config, Thread_Data.job, .{thread_data}) catch @panic("cannot create thread");
        }

        for (threads) |thread| {
            thread.join();
        }
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

    const num_codes: usize = @intCast(patient_id.dimensions[0]);
    if (cursor != num_codes) return null;

    return patient;
}

fn compute_c2c(id_1: u32, id_2: u32, _ontology_tree: []u32) f32 {
    if (id_1 == id_2) return 0;

    const res_1 = get_genealogy(id_1, _ontology_tree);
    const res_2 = get_genealogy(id_2, _ontology_tree);

    const genealogy_1 = res_1[0];
    const genealogy_2 = res_2[0];
    const root_1 = res_1[1];
    const root_2 = res_2[1];

    var cursor_1 = root_1;
    var cursor_2 = root_2;
    while (genealogy_1[cursor_1] == genealogy_2[cursor_2]) {
        if (cursor_1 == 0 or cursor_2 == 0) break;
        cursor_1 -= 1;
        cursor_2 -= 1;
    }
    cursor_1 = @min(cursor_1 + 1, root_1);
    cursor_2 = @min(cursor_2 + 1, root_2);

    const d_lr_doubled: f32 = @floatFromInt(2 * (root_1 - cursor_1));
    const dist = 1.0 - d_lr_doubled / (@as(f32, @floatFromInt(cursor_1 + cursor_2)) + d_lr_doubled);

    // @debug
    // if (dist < 1e-4) {
    //     std.debug.print("\n", .{});
    //     std.debug.print("id_1:{d: <8}id_2:{d: <8}\n", .{ id_1, id_2 });
    //     std.debug.print("genealogy_1 (root {d: <2} cursor {d: <2}): {any}\n", .{ root_1, cursor_1, genealogy_1 });
    //     std.debug.print("genealogy_1 (root {d: <2} cursor {d: <2}): {any}\n", .{ root_2, cursor_2, genealogy_2 });
    // }

    return dist;
}

fn get_genealogy(id: u32, _ontology_tree: []u32) struct { [genealogy_max_size]u32, usize } {
    var res = std.mem.zeroes([genealogy_max_size]u32);
    res[0] = id;
    var it: usize = 0;
    while (true) {
        const parent = _ontology_tree[res[it]];
        if (parent != res[it]) {
            it += 1;
            res[it] = parent;
        } else break;
    }
    return .{ res, it };
}

fn asymmetrical_v2v(v1: []u32, v2: []u32, _ontology: Ontology) f32 {
    var sum: f32 = 0;
    for (v1) |c1| {
        var best = std.math.floatMax(f32);
        for (v2) |c2| {
            const dist = _ontology.get(c1, c2);
            // const dist = blk: {
            //     var x: f32 = 1.0;
            //     if (c1 == c2) x = 0.0;
            //     break :blk x;
            // };
            best = @min(best, dist);
        }
        sum += best;
    }
    return sum;
}

fn compute_v2v(v1: []u32, v2: []u32, _ontology: Ontology) f32 {
    const x = asymmetrical_v2v(v1, v2, _ontology);
    const y = asymmetrical_v2v(v2, v1, _ontology);
    // return x + y;
    return @max(x, y);
}

fn compute_p2p(p1: [][]u32, p2: [][]u32, _ontology: Ontology, allocator: std.mem.Allocator) f32 {
    var table = allocator.alloc(f32, p1.len * p2.len) catch @panic("error with allocation");

    const w = p1.len;

    for (0..p1.len) |it| {
        for (0..p2.len) |jt| {
            const cost = compute_v2v(p1[it], p2[jt], _ontology);
            var in_cost: f32 = std.math.floatMax(f32);
            var del_cost: f32 = std.math.floatMax(f32);
            var edit_cost: f32 = std.math.floatMax(f32);

            if (it > 0) {
                in_cost = table[jt * w + it - 1];
                if (jt > 0) {
                    del_cost = table[(jt - 1) * w + it];
                    edit_cost = table[(jt - 1) * w + it - 1];
                }
            } else {
                if (jt > 0) {
                    del_cost = table[(jt - 1) * w + it];
                } else {
                    edit_cost = 0;
                }
            }

            table[jt * w + it] = cost + @min(in_cost, del_cost, edit_cost);
        }
    }

    // @debug
    // std.debug.print("\ntable {}x{}\n", .{ p1.len, p2.len });
    // for (0..p2.len) |jt| {
    //     for (0..p1.len) |it| {
    //         std.debug.print("{d:.1}  ", .{table[jt * w + it]});
    //     }
    //     std.debug.print("\n", .{});
    // }
    // std.debug.print("\n", .{});

    return table[table.len - 1];
}

/// result is a preallocated array for the result of the same size of dataset
fn find_neighbours(patient: [][]u32, dataset: [][][]u32, _ontology: Ontology, result: []f32) void {
    const Thread_Data = struct {
        ontology: Ontology,
        dataset: [][][]u32,
        patient: [][]u32,
        result: []f32,
        current_index: *usize,

        const Self = @This();
        const batch_size = 16;

        pub fn job(self: Self) void {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const allocator = arena.allocator();

            while (true) {
                const index: usize = @atomicRmw(usize, self.current_index, .Add, Self.batch_size, .Monotonic);
                if (index >= self.dataset.len) break;

                const index_limit = @min(self.dataset.len, index + batch_size);
                for (self.dataset[index..index_limit], index..) |d_patient, it| {
                    const dist = compute_p2p(self.patient, d_patient, self.ontology, allocator);
                    self.result[it] = dist;
                    _ = arena.reset(.retain_capacity);
                }
            }
        }
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const spawn_config = std.Thread.SpawnConfig{ .allocator = allocator };
    var threads = allocator.alloc(std.Thread, num_jobs) catch @panic("error in allocation");

    var current_index: usize = 0;

    for (threads) |*thread| {
        const thread_data = Thread_Data{
            .ontology = _ontology,
            .dataset = dataset,
            .patient = patient,
            .result = result,
            .current_index = &current_index,
        };
        thread.* = std.Thread.spawn(spawn_config, Thread_Data.job, .{thread_data}) catch @panic("cannot create thread");
    }

    for (threads) |thread| {
        thread.join();
    }
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
    const module = py.PyModule_Create(@constCast(&generator_module));
    if (module == null) return null;

    generator_error = py.PyErr_NewException("generator.error", null, null);
    py.Py_XINCREF(generator_error);
    if (py.PyModule_AddObject(module, "error", generator_error) < 0) {
        py.Py_XDECREF(generator_error);
        {
            const tmp = generator_error;
            if (tmp != null) {
                generator_error = null;
                py.Py_DECREF(tmp);
            }
        }
        py.Py_DECREF(module);
        return null;
    }

    np = import_numpy() catch std.debug.panic("cannot import numpy", .{});

    _arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    global_arena = _arena.allocator();

    {
        // @bug this numbers are dependent on the system they are running in
        // and they should be initialized in the _start function
        std.os.linux.tls.tls_image.alloc_size = 48;
        std.os.linux.tls.tls_image.alloc_align = 8;
        std.os.linux.tls.tls_image.tcb_offset = 16;
        std.os.linux.tls.tls_image.dtv_offset = 32;
        std.os.linux.tls.tls_image.data_offset = 0;
        std.os.linux.tls.tls_image.data_size = 16;
        std.os.linux.tls.tls_image.gdt_entry_number = 18446744073709551615;
    }

    return module;
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

const std = @import("std");
const py = @import("python.zig");

const Np = @import("numpy_data.zig");
