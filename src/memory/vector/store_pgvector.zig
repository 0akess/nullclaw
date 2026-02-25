//! PgvectorVectorStore — VectorStore vtable adapter for PostgreSQL with pgvector.
//!
//! Implements the VectorStore interface using pgvector's cosine distance
//! operator (<=>) for similarity search. Feature-gated behind
//! build_options.enable_postgres.
//!
//! SQL schema:
//!   CREATE EXTENSION IF NOT EXISTS vector;
//!   CREATE TABLE IF NOT EXISTS memory_vectors (
//!     key       TEXT PRIMARY KEY,
//!     embedding vector(N),
//!     updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
//!   );
//!   CREATE INDEX ON memory_vectors USING ivfflat (embedding vector_cosine_ops);

const std = @import("std");
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");
const store_mod = @import("store.zig");
const VectorStore = store_mod.VectorStore;
const VectorResult = store_mod.VectorResult;
const HealthStatus = store_mod.HealthStatus;

const c = if (build_options.enable_postgres) @cImport({
    @cInclude("libpq-fe.h");
}) else struct {};

// ── Config ────────────────────────────────────────────────────────

pub const PgvectorConfig = struct {
    connection_url: []const u8,
    table_name: []const u8 = "memory_vectors",
    dimensions: u32,
};

// ── PgvectorVectorStore ───────────────────────────────────────────

pub const PgvectorVectorStore = struct {
    allocator: Allocator,
    connection_url: []const u8,
    table_name: []const u8,
    dimensions: u32,
    conn: if (build_options.enable_postgres) ?*c.PGconn else void,
    owns_self: bool = false,

    const Self = @This();

    pub fn init(allocator: Allocator, config: PgvectorConfig) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .connection_url = try allocator.dupe(u8, config.connection_url),
            .table_name = try allocator.dupe(u8, config.table_name),
            .dimensions = config.dimensions,
            .conn = if (build_options.enable_postgres) null else {},
            .owns_self = true,
        };

        if (build_options.enable_postgres) {
            try self.connect();
            try self.ensureSchema();
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        const alloc = self.allocator;
        if (build_options.enable_postgres) {
            if (self.conn) |conn| c.PQfinish(conn);
        }
        alloc.free(self.connection_url);
        alloc.free(self.table_name);
        if (self.owns_self) alloc.destroy(self);
    }

    pub fn store(self: *Self) VectorStore {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable_instance,
        };
    }

    // ── Connection helpers ────────────────────────────────────────

    fn connect(self: *Self) !void {
        if (!build_options.enable_postgres) return;

        const url_z = try self.allocator.dupeZ(u8, self.connection_url);
        defer self.allocator.free(url_z);

        self.conn = c.PQconnectdb(url_z.ptr);
        if (self.conn) |conn| {
            if (c.PQstatus(conn) != c.CONNECTION_OK) {
                c.PQfinish(conn);
                self.conn = null;
                return error.PgConnectionFailed;
            }
        } else {
            return error.PgConnectionFailed;
        }
    }

    fn ensureSchema(self: *Self) !void {
        if (!build_options.enable_postgres) return;
        const conn = self.conn orelse return error.PgNotConnected;

        // Enable pgvector extension
        {
            const sql = "CREATE EXTENSION IF NOT EXISTS vector";
            const result = c.PQexec(conn, sql);
            defer c.PQclear(result);
            const status = c.PQresultStatus(result);
            if (status != c.PGRES_COMMAND_OK) return error.PgSchemaFailed;
        }

        // Create table with vector column
        const create_sql = try std.fmt.allocPrintZ(self.allocator,
            \\CREATE TABLE IF NOT EXISTS {s} (
            \\  key TEXT PRIMARY KEY,
            \\  embedding vector({d}),
            \\  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            \\)
        , .{ self.table_name, self.dimensions });
        defer self.allocator.free(create_sql);

        {
            const result = c.PQexec(conn, create_sql.ptr);
            defer c.PQclear(result);
            const status = c.PQresultStatus(result);
            if (status != c.PGRES_COMMAND_OK) return error.PgSchemaFailed;
        }
    }

    // ── Vector formatting helpers ─────────────────────────────────

    /// Format an f32 slice as a pgvector literal: "[0.1,0.2,0.3]"
    pub fn formatVector(allocator: Allocator, embedding: []const f32) ![]u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(allocator);

        try buf.append(allocator, '[');
        for (embedding, 0..) |val, i| {
            if (i > 0) try buf.append(allocator, ',');
            var tmp: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&tmp, "{d}", .{val}) catch continue;
            try buf.appendSlice(allocator, s);
        }
        try buf.append(allocator, ']');

        return allocator.dupe(u8, buf.items);
    }

    // ── VTable implementations ────────────────────────────────────

    fn implUpsert(ptr: *anyopaque, key: []const u8, embedding: []const f32) anyerror!void {
        if (!build_options.enable_postgres) return error.PgNotEnabled;
        const self: *Self = @ptrCast(@alignCast(ptr));
        const conn = self.conn orelse return error.PgNotConnected;
        const alloc = self.allocator;

        const vec_str = try formatVector(alloc, embedding);
        defer alloc.free(vec_str);

        const vec_z = try alloc.dupeZ(u8, vec_str);
        defer alloc.free(vec_z);

        const key_z = try alloc.dupeZ(u8, key);
        defer alloc.free(key_z);

        const sql = try std.fmt.allocPrintZ(alloc,
            "INSERT INTO {s} (key, embedding, updated_at) VALUES ($1, $2, now()) " ++
            "ON CONFLICT (key) DO UPDATE SET embedding = $2, updated_at = now()",
            .{self.table_name},
        );
        defer alloc.free(sql);

        const params = [_][*c]const u8{ key_z.ptr, vec_z.ptr };
        const result = c.PQexecParams(conn, sql.ptr, 2, null, &params, null, null, 0);
        defer c.PQclear(result);

        const status = c.PQresultStatus(result);
        if (status != c.PGRES_COMMAND_OK) return error.PgQueryFailed;
    }

    fn implSearch(ptr: *anyopaque, alloc: Allocator, query_embedding: []const f32, limit: u32) anyerror![]VectorResult {
        if (!build_options.enable_postgres) return error.PgNotEnabled;
        const self: *Self = @ptrCast(@alignCast(ptr));
        const conn = self.conn orelse return error.PgNotConnected;

        const vec_str = try formatVector(alloc, query_embedding);
        defer alloc.free(vec_str);

        const vec_z = try alloc.dupeZ(u8, vec_str);
        defer alloc.free(vec_z);

        var limit_buf: [16]u8 = undefined;
        const limit_str = try std.fmt.bufPrintZ(&limit_buf, "{d}", .{limit});

        // Use 1 - cosine_distance as similarity score
        const sql = try std.fmt.allocPrintZ(alloc,
            "SELECT key, 1 - (embedding <=> $1::vector) AS similarity " ++
            "FROM {s} ORDER BY embedding <=> $1::vector LIMIT $2",
            .{self.table_name},
        );
        defer alloc.free(sql);

        const params = [_][*c]const u8{ vec_z.ptr, limit_str.ptr };
        const result = c.PQexecParams(conn, sql.ptr, 2, null, &params, null, null, 0);
        defer c.PQclear(result);

        const status = c.PQresultStatus(result);
        if (status != c.PGRES_TUPLES_OK) return error.PgQueryFailed;

        const nrows = c.PQntuples(result);
        var results: std.ArrayListUnmanaged(VectorResult) = .empty;
        errdefer {
            for (results.items) |*r| r.deinit(alloc);
            results.deinit(alloc);
        }

        var row: c_int = 0;
        while (row < nrows) : (row += 1) {
            const key_raw = c.PQgetvalue(result, row, 0);
            const sim_raw = c.PQgetvalue(result, row, 1);

            if (key_raw == null or sim_raw == null) continue;

            const key_slice: []const u8 = std.mem.span(key_raw);
            const sim_slice: []const u8 = std.mem.span(sim_raw);

            const score = std.fmt.parseFloat(f32, sim_slice) catch 0.0;
            try results.append(alloc, .{
                .key = try alloc.dupe(u8, key_slice),
                .score = score,
            });
        }

        const out = try alloc.dupe(VectorResult, results.items);
        results.deinit(alloc);
        return out;
    }

    fn implDelete(ptr: *anyopaque, key: []const u8) anyerror!void {
        if (!build_options.enable_postgres) return error.PgNotEnabled;
        const self: *Self = @ptrCast(@alignCast(ptr));
        const conn = self.conn orelse return error.PgNotConnected;
        const alloc = self.allocator;

        const key_z = try alloc.dupeZ(u8, key);
        defer alloc.free(key_z);

        const sql = try std.fmt.allocPrintZ(alloc, "DELETE FROM {s} WHERE key = $1", .{self.table_name});
        defer alloc.free(sql);

        const params = [_][*c]const u8{key_z.ptr};
        const result = c.PQexecParams(conn, sql.ptr, 1, null, &params, null, null, 0);
        defer c.PQclear(result);

        const status = c.PQresultStatus(result);
        if (status != c.PGRES_COMMAND_OK) return error.PgQueryFailed;
    }

    fn implCount(ptr: *anyopaque) anyerror!usize {
        if (!build_options.enable_postgres) return error.PgNotEnabled;
        const self: *Self = @ptrCast(@alignCast(ptr));
        const conn = self.conn orelse return error.PgNotConnected;

        const sql = try std.fmt.allocPrintZ(self.allocator, "SELECT COUNT(*) FROM {s}", .{self.table_name});
        defer self.allocator.free(sql);

        const result = c.PQexec(conn, sql.ptr);
        defer c.PQclear(result);

        const status = c.PQresultStatus(result);
        if (status != c.PGRES_TUPLES_OK) return error.PgQueryFailed;

        if (c.PQntuples(result) < 1) return 0;

        const val_raw = c.PQgetvalue(result, 0, 0);
        if (val_raw == null) return 0;

        const val_slice: []const u8 = std.mem.span(val_raw);
        return std.fmt.parseInt(usize, val_slice, 10) catch 0;
    }

    fn implHealthCheck(ptr: *anyopaque, alloc: Allocator) anyerror!HealthStatus {
        if (!build_options.enable_postgres) {
            return HealthStatus{
                .ok = false,
                .latency_ns = 0,
                .entry_count = null,
                .error_msg = try alloc.dupe(u8, "pgvector not enabled"),
            };
        }

        const self: *Self = @ptrCast(@alignCast(ptr));
        const start = std.time.nanoTimestamp();

        const conn = self.conn orelse {
            const elapsed: u64 = @intCast(@max(0, std.time.nanoTimestamp() - start));
            return HealthStatus{
                .ok = false,
                .latency_ns = elapsed,
                .entry_count = null,
                .error_msg = try alloc.dupe(u8, "pgvector not connected"),
            };
        };

        const sql = "SELECT 1";
        const result = c.PQexec(conn, sql);
        defer c.PQclear(result);

        const elapsed: u64 = @intCast(@max(0, std.time.nanoTimestamp() - start));
        const status = c.PQresultStatus(result);

        if (status != c.PGRES_TUPLES_OK) {
            return HealthStatus{
                .ok = false,
                .latency_ns = elapsed,
                .entry_count = null,
                .error_msg = try alloc.dupe(u8, "pgvector health check failed"),
            };
        }

        // Best-effort count
        const entry_count: ?usize = blk: {
            const count_sql = try std.fmt.allocPrintZ(self.allocator, "SELECT COUNT(*) FROM {s}", .{self.table_name});
            defer self.allocator.free(count_sql);

            const count_result = c.PQexec(conn, count_sql.ptr);
            defer c.PQclear(count_result);

            if (c.PQresultStatus(count_result) != c.PGRES_TUPLES_OK) break :blk null;
            if (c.PQntuples(count_result) < 1) break :blk null;

            const val_raw = c.PQgetvalue(count_result, 0, 0);
            if (val_raw == null) break :blk null;

            const val_slice: []const u8 = std.mem.span(val_raw);
            break :blk std.fmt.parseInt(usize, val_slice, 10) catch null;
        };

        return HealthStatus{
            .ok = true,
            .latency_ns = elapsed,
            .entry_count = entry_count,
            .error_msg = null,
        };
    }

    fn implDeinit(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }

    const vtable_instance = VectorStore.VTable{
        .upsert = &implUpsert,
        .search = &implSearch,
        .delete = &implDelete,
        .count = &implCount,
        .health_check = &implHealthCheck,
        .deinit = &implDeinit,
    };
};

// ── Tests ─────────────────────────────────────────────────────────

test "formatVector basic" {
    const embedding = [_]f32{ 0.1, 0.2, 0.3 };
    const result = try PgvectorVectorStore.formatVector(std.testing.allocator, &embedding);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("[0.1,0.2,0.3]", result);
}

test "formatVector single element" {
    const embedding = [_]f32{1.5};
    const result = try PgvectorVectorStore.formatVector(std.testing.allocator, &embedding);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("[1.5]", result);
}

test "formatVector empty" {
    const empty: []const f32 = &.{};
    const result = try PgvectorVectorStore.formatVector(std.testing.allocator, empty);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("[]", result);
}

test "formatVector negative values" {
    const embedding = [_]f32{ -0.5, 1.0, -3.14 };
    const result = try PgvectorVectorStore.formatVector(std.testing.allocator, &embedding);
    defer std.testing.allocator.free(result);
    // Verify it starts with [ and ends with ]
    try std.testing.expect(result[0] == '[');
    try std.testing.expect(result[result.len - 1] == ']');
    // Verify it has 2 commas (3 elements)
    var comma_count: usize = 0;
    for (result) |ch| {
        if (ch == ',') comma_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), comma_count);
}

test "PgvectorVectorStore init and deinit without postgres" {
    if (build_options.enable_postgres) return;
    // When postgres is disabled, init should still create the struct
    // but won't connect
    const self = try PgvectorVectorStore.init(std.testing.allocator, .{
        .connection_url = "postgresql://localhost/test",
        .dimensions = 768,
    });
    try std.testing.expectEqualStrings("postgresql://localhost/test", self.connection_url);
    try std.testing.expectEqualStrings("memory_vectors", self.table_name);
    try std.testing.expectEqual(@as(u32, 768), self.dimensions);
    self.deinit();
}

test "PgvectorVectorStore produces valid VectorStore vtable" {
    if (build_options.enable_postgres) return;
    var self = try PgvectorVectorStore.init(std.testing.allocator, .{
        .connection_url = "postgresql://localhost/test",
        .dimensions = 384,
    });
    const s = self.store();
    try std.testing.expect(s.vtable.upsert == &PgvectorVectorStore.implUpsert);
    try std.testing.expect(s.vtable.search == &PgvectorVectorStore.implSearch);
    try std.testing.expect(s.vtable.delete == &PgvectorVectorStore.implDelete);
    try std.testing.expect(s.vtable.count == &PgvectorVectorStore.implCount);
    try std.testing.expect(s.vtable.health_check == &PgvectorVectorStore.implHealthCheck);
    try std.testing.expect(s.vtable.deinit == &PgvectorVectorStore.implDeinit);
    s.deinitStore();
}

test "PgvectorVectorStore healthCheck disabled returns not-ok" {
    if (build_options.enable_postgres) return;
    var self = try PgvectorVectorStore.init(std.testing.allocator, .{
        .connection_url = "postgresql://localhost/test",
        .dimensions = 768,
    });
    const s = self.store();
    defer s.deinitStore();

    const status = try s.healthCheck(std.testing.allocator);
    defer status.deinit(std.testing.allocator);

    try std.testing.expect(!status.ok);
    try std.testing.expect(status.error_msg != null);
}
