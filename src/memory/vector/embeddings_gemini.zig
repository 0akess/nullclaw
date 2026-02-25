//! Gemini embedding provider — Google's text-embedding-004 via embedContent API.
//!
//! API: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent
//! Auth: API key as query param `?key={api_key}`
//! Default model: "text-embedding-004", 768 dimensions

const std = @import("std");
const EmbeddingProvider = @import("embeddings.zig").EmbeddingProvider;

pub const GeminiEmbedding = struct {
    allocator: std.mem.Allocator,
    base_url: []const u8,
    api_key: []const u8,
    model: []const u8,
    dims: u32,

    const Self = @This();

    pub const default_base_url = "https://generativelanguage.googleapis.com";
    pub const default_model = "text-embedding-004";
    pub const default_dims: u32 = 768;

    pub fn init(
        allocator: std.mem.Allocator,
        api_key: []const u8,
        model: ?[]const u8,
        base_url: ?[]const u8,
        dims: ?u32,
    ) !*Self {
        const self_ = try allocator.create(Self);
        self_.* = .{
            .allocator = allocator,
            .base_url = try allocator.dupe(u8, base_url orelse default_base_url),
            .api_key = try allocator.dupe(u8, api_key),
            .model = try allocator.dupe(u8, model orelse default_model),
            .dims = dims orelse default_dims,
        };
        return self_;
    }

    pub fn deinitSelf(self: *Self) void {
        self.allocator.free(self.base_url);
        self.allocator.free(self.api_key);
        self.allocator.free(self.model);
        self.allocator.destroy(self);
    }

    fn buildUrl(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(
            allocator,
            "{s}/v1beta/models/{s}:embedContent?key={s}",
            .{ self.base_url, self.model, self.api_key },
        );
    }

    /// Build request body JSON: {"model":"models/{model}","content":{"parts":[{"text":"..."}]}}
    pub fn buildRequestBody(allocator: std.mem.Allocator, model: []const u8, text: []const u8) ![]u8 {
        var body_buf: std.ArrayListUnmanaged(u8) = .empty;
        defer body_buf.deinit(allocator);

        try body_buf.appendSlice(allocator, "{\"model\":\"models/");
        try appendJsonEscaped(&body_buf, allocator, model);
        try body_buf.appendSlice(allocator, "\",\"content\":{\"parts\":[{\"text\":\"");
        try appendJsonEscaped(&body_buf, allocator, text);
        try body_buf.appendSlice(allocator, "\"}]}}");

        return allocator.dupe(u8, body_buf.items);
    }

    fn implName(_: *anyopaque) []const u8 {
        return "gemini";
    }

    fn implDimensions(ptr: *anyopaque) u32 {
        const self_: *Self = @ptrCast(@alignCast(ptr));
        return self_.dims;
    }

    fn implEmbed(ptr: *anyopaque, allocator: std.mem.Allocator, text: []const u8) anyerror![]f32 {
        const self_: *Self = @ptrCast(@alignCast(ptr));

        if (text.len == 0) {
            return allocator.alloc(f32, 0);
        }

        const body = try buildRequestBody(allocator, self_.model, text);
        defer allocator.free(body);

        const url = try self_.buildUrl(allocator);
        defer allocator.free(url);

        var client = std.http.Client{ .allocator = allocator };
        defer client.deinit();

        var aw: std.Io.Writer.Allocating = .init(allocator);
        defer aw.deinit();

        const result = client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = body,
            .extra_headers = &.{
                .{ .name = "Content-Type", .value = "application/json" },
            },
            .response_writer = &aw.writer,
        }) catch return error.EmbeddingApiError;

        if (result.status != .ok) {
            return error.EmbeddingApiError;
        }

        const resp_body = aw.writer.buffer[0..aw.writer.end];
        if (resp_body.len == 0) return error.EmbeddingApiError;

        return parseGeminiResponse(allocator, resp_body);
    }

    fn implDeinit(ptr: *anyopaque) void {
        const self_: *Self = @ptrCast(@alignCast(ptr));
        self_.deinitSelf();
    }

    const vtable = EmbeddingProvider.VTable{
        .name = &implName,
        .dimensions = &implDimensions,
        .embed = &implEmbed,
        .deinit = &implDeinit,
    };

    pub fn provider(self: *Self) EmbeddingProvider {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }
};

/// Parse Gemini response: {"embedding":{"values":[0.1, 0.2, ...]}}
pub fn parseGeminiResponse(allocator: std.mem.Allocator, json_bytes: []const u8) ![]f32 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch return error.InvalidEmbeddingResponse;
    defer parsed.deinit();

    const root = parsed.value;
    const embedding_obj = root.object.get("embedding") orelse return error.InvalidEmbeddingResponse;
    const values = switch (embedding_obj) {
        .object => |obj| obj.get("values") orelse return error.InvalidEmbeddingResponse,
        else => return error.InvalidEmbeddingResponse,
    };
    const arr = switch (values) {
        .array => |a| a,
        else => return error.InvalidEmbeddingResponse,
    };

    const result = try allocator.alloc(f32, arr.items.len);
    for (arr.items, 0..) |val, i| {
        result[i] = switch (val) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => 0.0,
        };
    }
    return result;
}

fn appendJsonEscaped(buf: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, text: []const u8) !void {
    for (text) |ch| {
        switch (ch) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => {
                if (ch < 0x20) {
                    var hex_buf: [6]u8 = undefined;
                    const hex = std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{ch}) catch continue;
                    try buf.appendSlice(allocator, hex);
                } else {
                    try buf.append(allocator, ch);
                }
            },
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "GeminiEmbedding init and deinit" {
    var impl_ = try GeminiEmbedding.init(
        std.testing.allocator,
        "test-api-key",
        null,
        null,
        null,
    );
    const p = impl_.provider();
    try std.testing.expectEqualStrings("gemini", p.getName());
    try std.testing.expectEqual(@as(u32, 768), p.getDimensions());
    p.deinit();
}

test "GeminiEmbedding init with custom values" {
    var impl_ = try GeminiEmbedding.init(
        std.testing.allocator,
        "my-key",
        "gemini-embedding-001",
        "https://custom.api.example.com",
        1024,
    );
    const p = impl_.provider();
    try std.testing.expectEqualStrings("gemini", p.getName());
    try std.testing.expectEqual(@as(u32, 1024), p.getDimensions());
    p.deinit();
}

test "GeminiEmbedding embed empty text" {
    var impl_ = try GeminiEmbedding.init(
        std.testing.allocator,
        "test-key",
        null,
        null,
        null,
    );
    const p = impl_.provider();
    defer p.deinit();

    const vec = try p.embed(std.testing.allocator, "");
    defer std.testing.allocator.free(vec);
    try std.testing.expectEqual(@as(usize, 0), vec.len);
}

test "GeminiEmbedding buildRequestBody" {
    const body = try GeminiEmbedding.buildRequestBody(std.testing.allocator, "text-embedding-004", "hello world");
    defer std.testing.allocator.free(body);

    // Verify it's valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, body, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const model_val = root.object.get("model") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("models/text-embedding-004", model_val.string);

    const content = root.object.get("content") orelse return error.TestFailed;
    const parts = content.object.get("parts") orelse return error.TestFailed;
    const first_part = parts.array.items[0];
    const text_val = first_part.object.get("text") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("hello world", text_val.string);
}

test "GeminiEmbedding buildRequestBody escapes special chars" {
    const body = try GeminiEmbedding.buildRequestBody(std.testing.allocator, "model", "hello \"world\"\nnewline");
    defer std.testing.allocator.free(body);

    // Verify it's valid JSON (escaped properly)
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, body, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const content = root.object.get("content") orelse return error.TestFailed;
    const parts = content.object.get("parts") orelse return error.TestFailed;
    const first_part = parts.array.items[0];
    const text_val = first_part.object.get("text") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("hello \"world\"\nnewline", text_val.string);
}

test "parseGeminiResponse valid" {
    const json =
        \\{"embedding":{"values":[0.1,0.2,0.3]}}
    ;
    const result = try parseGeminiResponse(std.testing.allocator, json);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expect(@abs(result[0] - 0.1) < 0.001);
    try std.testing.expect(@abs(result[1] - 0.2) < 0.001);
    try std.testing.expect(@abs(result[2] - 0.3) < 0.001);
}

test "parseGeminiResponse empty values" {
    const json =
        \\{"embedding":{"values":[]}}
    ;
    const result = try parseGeminiResponse(std.testing.allocator, json);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "parseGeminiResponse missing embedding" {
    const json =
        \\{"error":"bad request"}
    ;
    const result = parseGeminiResponse(std.testing.allocator, json);
    try std.testing.expectError(error.InvalidEmbeddingResponse, result);
}

test "parseGeminiResponse missing values" {
    const json =
        \\{"embedding":{}}
    ;
    const result = parseGeminiResponse(std.testing.allocator, json);
    try std.testing.expectError(error.InvalidEmbeddingResponse, result);
}

test "parseGeminiResponse integer values" {
    const json =
        \\{"embedding":{"values":[1,2,3]}}
    ;
    const result = try parseGeminiResponse(std.testing.allocator, json);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expect(@abs(result[0] - 1.0) < 0.001);
}
