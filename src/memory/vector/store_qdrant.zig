//! QdrantVectorStore — VectorStore vtable adapter for Qdrant REST API.
//!
//! Implements the VectorStore interface by sending HTTP requests to a
//! running Qdrant instance. Supports upsert, search, delete, count,
//! and healthCheck via the Qdrant REST endpoints.

const std = @import("std");
const Allocator = std.mem.Allocator;
const store_mod = @import("store.zig");
const VectorStore = store_mod.VectorStore;
const VectorResult = store_mod.VectorResult;
const HealthStatus = store_mod.HealthStatus;

// ── Config ────────────────────────────────────────────────────────

pub const QdrantConfig = struct {
    url: []const u8, // e.g. "http://localhost:6333"
    api_key: ?[]const u8, // optional
    collection_name: []const u8, // e.g. "nullclaw_memories"
    dimensions: u32, // must match embedding provider
};

// ── QdrantVectorStore ─────────────────────────────────────────────

pub const QdrantVectorStore = struct {
    allocator: Allocator,
    url: []const u8,
    api_key: ?[]const u8,
    collection_name: []const u8,
    dimensions: u32,
    owns_self: bool = false,

    const Self = @This();

    pub fn init(allocator: Allocator, config: QdrantConfig) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .url = try allocator.dupe(u8, config.url),
            .api_key = if (config.api_key) |k| try allocator.dupe(u8, k) else null,
            .collection_name = try allocator.dupe(u8, config.collection_name),
            .dimensions = config.dimensions,
            .owns_self = true,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        const alloc = self.allocator;
        alloc.free(self.url);
        if (self.api_key) |k| alloc.free(k);
        alloc.free(self.collection_name);
        if (self.owns_self) alloc.destroy(self);
    }

    pub fn store(self: *Self) VectorStore {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable_instance,
        };
    }

    // ── HTTP helpers ──────────────────────────────────────────────

    fn buildUrl(self: *const Self, alloc: Allocator, path: []const u8) ![]u8 {
        return std.fmt.allocPrint(alloc, "{s}/collections/{s}{s}", .{
            self.url,
            self.collection_name,
            path,
        });
    }

    fn doRequest(
        self: *const Self,
        alloc: Allocator,
        url: []const u8,
        method: std.http.Method,
        payload: ?[]const u8,
    ) !struct { status: std.http.Status, body: []u8 } {
        var client = std.http.Client{ .allocator = alloc };
        defer client.deinit();

        var aw: std.Io.Writer.Allocating = .init(alloc);
        errdefer aw.deinit();

        var extra_headers_buf: [3]std.http.Header = undefined;
        var header_count: usize = 0;

        extra_headers_buf[header_count] = .{ .name = "Content-Type", .value = "application/json" };
        header_count += 1;

        var auth_header: ?[]u8 = null;
        defer if (auth_header) |h| alloc.free(h);

        if (self.api_key) |key| {
            auth_header = try std.fmt.allocPrint(alloc, "Bearer {s}", .{key});
            extra_headers_buf[header_count] = .{ .name = "api-key", .value = auth_header.? };
            header_count += 1;
        }

        const result = client.fetch(.{
            .location = .{ .url = url },
            .method = method,
            .payload = payload,
            .extra_headers = extra_headers_buf[0..header_count],
            .response_writer = &aw.writer,
        }) catch return error.QdrantConnectionError;

        const body = try alloc.dupe(u8, aw.writer.buffer[0..aw.writer.end]);
        aw.deinit();

        return .{ .status = result.status, .body = body };
    }

    // ── JSON builders ─────────────────────────────────────────────

    fn buildUpsertPayload(alloc: Allocator, key: []const u8, embedding: []const f32) ![]u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(alloc);

        try buf.appendSlice(alloc, "{\"points\":[{\"id\":\"");
        try appendJsonEscaped(&buf, alloc, key);
        try buf.appendSlice(alloc, "\",\"vector\":[");
        for (embedding, 0..) |val, i| {
            if (i > 0) try buf.append(alloc, ',');
            var tmp: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&tmp, "{d}", .{val}) catch continue;
            try buf.appendSlice(alloc, s);
        }
        try buf.appendSlice(alloc, "],\"payload\":{\"key\":\"");
        try appendJsonEscaped(&buf, alloc, key);
        try buf.appendSlice(alloc, "\"}}]}");

        return alloc.dupe(u8, buf.items);
    }

    fn buildSearchPayload(alloc: Allocator, query_embedding: []const f32, limit: u32) ![]u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(alloc);

        try buf.appendSlice(alloc, "{\"vector\":[");
        for (query_embedding, 0..) |val, i| {
            if (i > 0) try buf.append(alloc, ',');
            var tmp: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&tmp, "{d}", .{val}) catch continue;
            try buf.appendSlice(alloc, s);
        }
        try buf.appendSlice(alloc, "],\"limit\":");
        var lim_buf: [16]u8 = undefined;
        const lim_str = std.fmt.bufPrint(&lim_buf, "{d}", .{limit}) catch "10";
        try buf.appendSlice(alloc, lim_str);
        try buf.appendSlice(alloc, ",\"with_payload\":true}");

        return alloc.dupe(u8, buf.items);
    }

    fn buildDeletePayload(alloc: Allocator, key: []const u8) ![]u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(alloc);

        try buf.appendSlice(alloc, "{\"filter\":{\"must\":[{\"key\":\"key\",\"match\":{\"value\":\"");
        try appendJsonEscaped(&buf, alloc, key);
        try buf.appendSlice(alloc, "\"}}]}}");

        return alloc.dupe(u8, buf.items);
    }

    fn appendJsonEscaped(buf: *std.ArrayListUnmanaged(u8), alloc: Allocator, text: []const u8) !void {
        for (text) |ch| {
            switch (ch) {
                '"' => try buf.appendSlice(alloc, "\\\""),
                '\\' => try buf.appendSlice(alloc, "\\\\"),
                '\n' => try buf.appendSlice(alloc, "\\n"),
                '\r' => try buf.appendSlice(alloc, "\\r"),
                '\t' => try buf.appendSlice(alloc, "\\t"),
                else => {
                    if (ch < 0x20) {
                        var hex_buf: [6]u8 = undefined;
                        const hex = std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{ch}) catch continue;
                        try buf.appendSlice(alloc, hex);
                    } else {
                        try buf.append(alloc, ch);
                    }
                },
            }
        }
    }

    // ── Response parsers ──────────────────────────────────────────

    fn parseSearchResults(alloc: Allocator, body: []const u8) ![]VectorResult {
        const parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return error.QdrantInvalidResponse;
        defer parsed.deinit();

        const root = parsed.value;
        const result_arr = switch (root) {
            .object => |obj| blk: {
                const r = obj.get("result") orelse return error.QdrantInvalidResponse;
                break :blk switch (r) {
                    .array => |a| a,
                    else => return error.QdrantInvalidResponse,
                };
            },
            else => return error.QdrantInvalidResponse,
        };

        var results: std.ArrayListUnmanaged(VectorResult) = .empty;
        errdefer {
            for (results.items) |*r| r.deinit(alloc);
            results.deinit(alloc);
        }

        for (result_arr.items) |item| {
            const obj = switch (item) {
                .object => |o| o,
                else => continue,
            };

            // Extract score
            const score_val = obj.get("score") orelse continue;
            const score: f32 = switch (score_val) {
                .float => |f| @floatCast(f),
                .integer => |n| @floatFromInt(n),
                else => continue,
            };

            // Extract key from payload
            const payload_val = obj.get("payload") orelse continue;
            const payload = switch (payload_val) {
                .object => |o| o,
                else => continue,
            };
            const key_val = payload.get("key") orelse continue;
            const key_str = switch (key_val) {
                .string => |s| s,
                else => continue,
            };

            try results.append(alloc, .{
                .key = try alloc.dupe(u8, key_str),
                .score = score,
            });
        }

        const out = try alloc.dupe(VectorResult, results.items);
        results.deinit(alloc);
        return out;
    }

    fn parseCountResult(alloc: Allocator, body: []const u8) !usize {
        const parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return error.QdrantInvalidResponse;
        defer parsed.deinit();

        const root = parsed.value;
        const result_obj = switch (root) {
            .object => |obj| blk: {
                const r = obj.get("result") orelse return error.QdrantInvalidResponse;
                break :blk switch (r) {
                    .object => |o| o,
                    else => return error.QdrantInvalidResponse,
                };
            },
            else => return error.QdrantInvalidResponse,
        };

        const count_val = result_obj.get("count") orelse return error.QdrantInvalidResponse;
        return switch (count_val) {
            .integer => |n| @intCast(n),
            else => return error.QdrantInvalidResponse,
        };
    }

    // ── VTable implementations ────────────────────────────────────

    fn implUpsert(ptr: *anyopaque, key: []const u8, embedding: []const f32) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        const alloc = self.allocator;

        const url = try self.buildUrl(alloc, "/points?wait=true");
        defer alloc.free(url);

        const payload = try buildUpsertPayload(alloc, key, embedding);
        defer alloc.free(payload);

        const resp = try self.doRequest(alloc, url, .PUT, payload);
        defer alloc.free(resp.body);

        if (resp.status != .ok) return error.QdrantApiError;
    }

    fn implSearch(ptr: *anyopaque, alloc: Allocator, query_embedding: []const f32, limit: u32) anyerror![]VectorResult {
        const self: *Self = @ptrCast(@alignCast(ptr));

        const url = try self.buildUrl(alloc, "/points/search");
        defer alloc.free(url);

        const payload = try buildSearchPayload(alloc, query_embedding, limit);
        defer alloc.free(payload);

        const resp = try self.doRequest(alloc, url, .POST, payload);
        defer alloc.free(resp.body);

        if (resp.status != .ok) return error.QdrantApiError;

        return parseSearchResults(alloc, resp.body);
    }

    fn implDelete(ptr: *anyopaque, key: []const u8) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        const alloc = self.allocator;

        const url = try self.buildUrl(alloc, "/points/delete?wait=true");
        defer alloc.free(url);

        const payload = try buildDeletePayload(alloc, key);
        defer alloc.free(payload);

        const resp = try self.doRequest(alloc, url, .POST, payload);
        defer alloc.free(resp.body);

        if (resp.status != .ok) return error.QdrantApiError;
    }

    fn implCount(ptr: *anyopaque) anyerror!usize {
        const self: *Self = @ptrCast(@alignCast(ptr));
        const alloc = self.allocator;

        const url = try self.buildUrl(alloc, "/points/count");
        defer alloc.free(url);

        const payload = "{\"exact\":true}";
        const resp = try self.doRequest(alloc, url, .POST, payload);
        defer alloc.free(resp.body);

        if (resp.status != .ok) return error.QdrantApiError;

        return parseCountResult(alloc, resp.body);
    }

    fn implHealthCheck(ptr: *anyopaque, alloc: Allocator) anyerror!HealthStatus {
        const self: *Self = @ptrCast(@alignCast(ptr));
        const start = std.time.nanoTimestamp();

        // Hit the Qdrant healthz endpoint (not collection-scoped)
        const url = try std.fmt.allocPrint(alloc, "{s}/healthz", .{self.url});
        defer alloc.free(url);

        var client = std.http.Client{ .allocator = alloc };
        defer client.deinit();

        var aw: std.Io.Writer.Allocating = .init(alloc);
        defer aw.deinit();

        const result = client.fetch(.{
            .location = .{ .url = url },
            .method = .GET,
            .response_writer = &aw.writer,
        }) catch {
            const elapsed: u64 = @intCast(@max(0, std.time.nanoTimestamp() - start));
            return HealthStatus{
                .ok = false,
                .latency_ns = elapsed,
                .entry_count = null,
                .error_msg = try alloc.dupe(u8, "qdrant connection failed"),
            };
        };

        const elapsed: u64 = @intCast(@max(0, std.time.nanoTimestamp() - start));

        if (result.status != .ok) {
            return HealthStatus{
                .ok = false,
                .latency_ns = elapsed,
                .entry_count = null,
                .error_msg = try alloc.dupe(u8, "qdrant healthz returned non-200"),
            };
        }

        // Optionally get entry count (best-effort)
        const entry_count: ?usize = self.implCountInternal() catch null;

        return HealthStatus{
            .ok = true,
            .latency_ns = elapsed,
            .entry_count = entry_count,
            .error_msg = null,
        };
    }

    fn implCountInternal(self: *Self) !usize {
        const alloc = self.allocator;
        const url = try self.buildUrl(alloc, "/points/count");
        defer alloc.free(url);

        const payload = "{\"exact\":true}";
        const resp = try self.doRequest(alloc, url, .POST, payload);
        defer alloc.free(resp.body);

        if (resp.status != .ok) return error.QdrantApiError;

        return parseCountResult(alloc, resp.body);
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

test "QdrantVectorStore init and deinit" {
    const q = try QdrantVectorStore.init(std.testing.allocator, .{
        .url = "http://localhost:6333",
        .api_key = "test-key",
        .collection_name = "test_collection",
        .dimensions = 384,
    });
    // Verify fields are duped (not pointers to stack memory)
    try std.testing.expectEqualStrings("http://localhost:6333", q.url);
    try std.testing.expectEqualStrings("test-key", q.api_key.?);
    try std.testing.expectEqualStrings("test_collection", q.collection_name);
    try std.testing.expectEqual(@as(u32, 384), q.dimensions);
    q.deinit();
}

test "QdrantVectorStore init without api_key" {
    const q = try QdrantVectorStore.init(std.testing.allocator, .{
        .url = "http://localhost:6333",
        .api_key = null,
        .collection_name = "nullclaw",
        .dimensions = 1536,
    });
    try std.testing.expectEqual(@as(?[]const u8, null), q.api_key);
    q.deinit();
}

test "QdrantVectorStore produces valid VectorStore vtable" {
    var q = try QdrantVectorStore.init(std.testing.allocator, .{
        .url = "http://localhost:6333",
        .api_key = null,
        .collection_name = "test",
        .dimensions = 3,
    });
    const s = q.store();
    // Verify the vtable is wired correctly (methods are non-null function pointers)
    try std.testing.expect(s.vtable.upsert == &QdrantVectorStore.implUpsert);
    try std.testing.expect(s.vtable.search == &QdrantVectorStore.implSearch);
    try std.testing.expect(s.vtable.delete == &QdrantVectorStore.implDelete);
    try std.testing.expect(s.vtable.count == &QdrantVectorStore.implCount);
    try std.testing.expect(s.vtable.health_check == &QdrantVectorStore.implHealthCheck);
    try std.testing.expect(s.vtable.deinit == &QdrantVectorStore.implDeinit);
    s.deinitStore();
}

test "buildUpsertPayload generates valid JSON" {
    const alloc = std.testing.allocator;
    const embedding = [_]f32{ 0.1, 0.2, 0.3 };
    const payload = try QdrantVectorStore.buildUpsertPayload(alloc, "test_key", &embedding);
    defer alloc.free(payload);

    // Parse the JSON to verify it's valid
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, payload, .{});
    defer parsed.deinit();

    // Verify structure: {"points":[{"id":"test_key","vector":[...],"payload":{"key":"test_key"}}]}
    const points = parsed.value.object.get("points").?.array;
    try std.testing.expectEqual(@as(usize, 1), points.items.len);

    const point = points.items[0].object;
    try std.testing.expectEqualStrings("test_key", point.get("id").?.string);

    const vec = point.get("vector").?.array;
    try std.testing.expectEqual(@as(usize, 3), vec.items.len);

    const pl = point.get("payload").?.object;
    try std.testing.expectEqualStrings("test_key", pl.get("key").?.string);
}

test "buildUpsertPayload escapes special characters" {
    const alloc = std.testing.allocator;
    const embedding = [_]f32{1.0};
    const payload = try QdrantVectorStore.buildUpsertPayload(alloc, "key\"with\\quotes", &embedding);
    defer alloc.free(payload);

    // Should be valid JSON despite special chars in key
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, payload, .{});
    defer parsed.deinit();

    const points = parsed.value.object.get("points").?.array;
    const point = points.items[0].object;
    try std.testing.expectEqualStrings("key\"with\\quotes", point.get("id").?.string);
}

test "buildSearchPayload generates valid JSON" {
    const alloc = std.testing.allocator;
    const query = [_]f32{ 1.0, 2.0 };
    const payload = try QdrantVectorStore.buildSearchPayload(alloc, &query, 5);
    defer alloc.free(payload);

    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, payload, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    const vec = root.get("vector").?.array;
    try std.testing.expectEqual(@as(usize, 2), vec.items.len);

    const limit = root.get("limit").?.integer;
    try std.testing.expectEqual(@as(i64, 5), limit);

    try std.testing.expect(root.get("with_payload").?.bool);
}

test "buildDeletePayload generates valid JSON" {
    const alloc = std.testing.allocator;
    const payload = try QdrantVectorStore.buildDeletePayload(alloc, "my_key");
    defer alloc.free(payload);

    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, payload, .{});
    defer parsed.deinit();

    // {"filter":{"must":[{"key":"key","match":{"value":"my_key"}}]}}
    const filter = parsed.value.object.get("filter").?.object;
    const must = filter.get("must").?.array;
    try std.testing.expectEqual(@as(usize, 1), must.items.len);

    const cond = must.items[0].object;
    try std.testing.expectEqualStrings("key", cond.get("key").?.string);
    const match = cond.get("match").?.object;
    try std.testing.expectEqualStrings("my_key", match.get("value").?.string);
}

test "parseSearchResults valid response" {
    const alloc = std.testing.allocator;
    const json =
        \\{"result":[{"id":"abc","score":0.95,"payload":{"key":"mem_1"}},{"id":"def","score":0.8,"payload":{"key":"mem_2"}}],"status":"ok"}
    ;

    const results = try QdrantVectorStore.parseSearchResults(alloc, json);
    defer {
        for (results) |*r| r.deinit(alloc);
        alloc.free(results);
    }

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("mem_1", results[0].key);
    try std.testing.expect(@abs(results[0].score - 0.95) < 0.01);
    try std.testing.expectEqualStrings("mem_2", results[1].key);
    try std.testing.expect(@abs(results[1].score - 0.8) < 0.01);
}

test "parseSearchResults empty result" {
    const alloc = std.testing.allocator;
    const json =
        \\{"result":[],"status":"ok"}
    ;

    const results = try QdrantVectorStore.parseSearchResults(alloc, json);
    defer alloc.free(results);

    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "parseSearchResults invalid JSON returns error" {
    const alloc = std.testing.allocator;
    const result = QdrantVectorStore.parseSearchResults(alloc, "not json");
    try std.testing.expectError(error.QdrantInvalidResponse, result);
}

test "parseSearchResults missing result field returns error" {
    const alloc = std.testing.allocator;
    const json =
        \\{"status":"ok"}
    ;
    const result = QdrantVectorStore.parseSearchResults(alloc, json);
    try std.testing.expectError(error.QdrantInvalidResponse, result);
}

test "parseCountResult valid response" {
    const alloc = std.testing.allocator;
    const json =
        \\{"result":{"count":42},"status":"ok"}
    ;

    const count = try QdrantVectorStore.parseCountResult(alloc, json);
    try std.testing.expectEqual(@as(usize, 42), count);
}

test "parseCountResult zero count" {
    const alloc = std.testing.allocator;
    const json =
        \\{"result":{"count":0},"status":"ok"}
    ;

    const count = try QdrantVectorStore.parseCountResult(alloc, json);
    try std.testing.expectEqual(@as(usize, 0), count);
}

test "parseCountResult invalid JSON returns error" {
    const alloc = std.testing.allocator;
    const result = QdrantVectorStore.parseCountResult(alloc, "bad");
    try std.testing.expectError(error.QdrantInvalidResponse, result);
}

test "parseCountResult missing count field returns error" {
    const alloc = std.testing.allocator;
    const json =
        \\{"result":{},"status":"ok"}
    ;
    const result = QdrantVectorStore.parseCountResult(alloc, json);
    try std.testing.expectError(error.QdrantInvalidResponse, result);
}

test "appendJsonEscaped handles special characters" {
    const alloc = std.testing.allocator;
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(alloc);

    try QdrantVectorStore.appendJsonEscaped(&buf, alloc, "hello \"world\"\nnewline\\slash");

    try std.testing.expectEqualStrings("hello \\\"world\\\"\\nnewline\\\\slash", buf.items);
}

test "appendJsonEscaped handles plain text" {
    const alloc = std.testing.allocator;
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(alloc);

    try QdrantVectorStore.appendJsonEscaped(&buf, alloc, "simple text");

    try std.testing.expectEqualStrings("simple text", buf.items);
}

test "buildUrl constructs correct endpoint" {
    var q = try QdrantVectorStore.init(std.testing.allocator, .{
        .url = "http://localhost:6333",
        .api_key = null,
        .collection_name = "my_coll",
        .dimensions = 3,
    });
    defer q.deinit();

    const url = try q.buildUrl(std.testing.allocator, "/points/search");
    defer std.testing.allocator.free(url);

    try std.testing.expectEqualStrings("http://localhost:6333/collections/my_coll/points/search", url);
}

test "buildUrl with empty path" {
    var q = try QdrantVectorStore.init(std.testing.allocator, .{
        .url = "http://localhost:6333",
        .api_key = null,
        .collection_name = "test",
        .dimensions = 3,
    });
    defer q.deinit();

    const url = try q.buildUrl(std.testing.allocator, "");
    defer std.testing.allocator.free(url);

    try std.testing.expectEqualStrings("http://localhost:6333/collections/test", url);
}
