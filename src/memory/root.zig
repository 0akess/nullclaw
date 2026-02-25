//! Memory module — persistent knowledge storage for nullclaw.
//!
//! Mirrors ZeroClaw's memory architecture:
//!   - Memory vtable interface (store, recall, get, list, forget, count)
//!   - MemoryEntry, MemoryCategory
//!   - Multiple backends: SQLite (FTS5), Markdown (file-based), None (no-op)
//!   - ResponseCache for LLM response deduplication
//!   - Document chunking for large markdown files

const std = @import("std");
const build_options = @import("build_options");
const config_types = @import("../config_types.zig");
const log = std.log.scoped(.memory);

// engines/ (Layer A: Primary Store)
pub const sqlite = @import("engines/sqlite.zig");
pub const markdown = @import("engines/markdown.zig");
pub const none = @import("engines/none.zig");
pub const memory_lru = @import("engines/memory_lru.zig");
pub const lucid = @import("engines/lucid.zig");
pub const postgres = if (build_options.enable_postgres) @import("engines/postgres.zig") else struct {};
pub const redis = @import("engines/redis.zig");
pub const lancedb = @import("engines/lancedb.zig");
pub const registry = @import("engines/registry.zig");

// retrieval/ (Layer B: Retrieval Engine)
pub const retrieval = @import("retrieval/engine.zig");
pub const retrieval_qmd = @import("retrieval/qmd.zig");
pub const rrf = @import("retrieval/rrf.zig");
pub const query_expansion = @import("retrieval/query_expansion.zig");
pub const temporal_decay = @import("retrieval/temporal_decay.zig");
pub const mmr = @import("retrieval/mmr.zig");
pub const adaptive = @import("retrieval/adaptive.zig");
pub const llm_reranker = @import("retrieval/llm_reranker.zig");

// vector/ (Layer C: Vector Plane)
pub const vector = @import("vector/math.zig");
pub const vector_store = @import("vector/store.zig");
pub const embeddings = @import("vector/embeddings.zig");
pub const embeddings_gemini = @import("vector/embeddings_gemini.zig");
pub const embeddings_voyage = @import("vector/embeddings_voyage.zig");
pub const embeddings_ollama = @import("vector/embeddings_ollama.zig");
pub const provider_router = @import("vector/provider_router.zig");
pub const store_qdrant = @import("vector/store_qdrant.zig");
pub const store_pgvector = @import("vector/store_pgvector.zig");
pub const circuit_breaker = @import("vector/circuit_breaker.zig");
pub const outbox = @import("vector/outbox.zig");
pub const chunker = @import("vector/chunker.zig");

// lifecycle/ (Layer D: Runtime Orchestrator)
pub const cache = @import("lifecycle/cache.zig");
pub const semantic_cache = @import("lifecycle/semantic_cache.zig");
pub const hygiene = @import("lifecycle/hygiene.zig");
pub const snapshot = @import("lifecycle/snapshot.zig");
pub const rollout = @import("lifecycle/rollout.zig");
pub const migrate = @import("lifecycle/migrate.zig");
pub const diagnostics = @import("lifecycle/diagnostics.zig");
pub const summarizer = @import("lifecycle/summarizer.zig");

pub const SqliteMemory = sqlite.SqliteMemory;
pub const MarkdownMemory = markdown.MarkdownMemory;
pub const NoneMemory = none.NoneMemory;
pub const InMemoryLruMemory = memory_lru.InMemoryLruMemory;
pub const LucidMemory = lucid.LucidMemory;
pub const PostgresMemory = if (build_options.enable_postgres) postgres.PostgresMemory else struct {};
pub const RedisMemory = redis.RedisMemory;
pub const LanceDbMemory = lancedb.LanceDbMemory;
pub const ResponseCache = cache.ResponseCache;
pub const Chunk = chunker.Chunk;
pub const chunkMarkdown = chunker.chunkMarkdown;
pub const EmbeddingProvider = embeddings.EmbeddingProvider;
pub const NoopEmbedding = embeddings.NoopEmbedding;
pub const cosineSimilarity = vector.cosineSimilarity;
pub const ScoredResult = vector.ScoredResult;
pub const hybridMerge = vector.hybridMerge;
pub const HygieneReport = hygiene.HygieneReport;
pub const exportSnapshot = snapshot.exportSnapshot;
pub const hydrateFromSnapshot = snapshot.hydrateFromSnapshot;
pub const shouldHydrate = snapshot.shouldHydrate;
pub const BackendDescriptor = registry.BackendDescriptor;
pub const BackendConfig = registry.BackendConfig;
pub const BackendInstance = registry.BackendInstance;
pub const BackendCapabilities = registry.BackendCapabilities;
pub const findBackend = registry.findBackend;
pub const RetrievalCandidate = retrieval.RetrievalCandidate;
pub const RetrievalSourceAdapter = retrieval.RetrievalSourceAdapter;
pub const PrimaryAdapter = retrieval.PrimaryAdapter;
pub const RetrievalEngine = retrieval.RetrievalEngine;
pub const QmdAdapter = retrieval_qmd.QmdAdapter;
pub const rrfMerge = rrf.rrfMerge;
pub const applyTemporalDecay = temporal_decay.applyTemporalDecay;
pub const VectorStore = vector_store.VectorStore;
pub const VectorResult = vector_store.VectorResult;
pub const HealthStatus = vector_store.HealthStatus;
pub const SqliteSharedVectorStore = vector_store.SqliteSharedVectorStore;
pub const QdrantVectorStore = store_qdrant.QdrantVectorStore;
pub const freeVectorResults = vector_store.freeVectorResults;
pub const VectorOutbox = outbox.VectorOutbox;
pub const CircuitBreaker = circuit_breaker.CircuitBreaker;
pub const RolloutMode = rollout.RolloutMode;
pub const RolloutPolicy = rollout.RolloutPolicy;
pub const RolloutDecision = rollout.RolloutDecision;
pub const SqliteSourceEntry = migrate.SqliteSourceEntry;
pub const readBrainDb = migrate.readBrainDb;
pub const freeSqliteEntries = migrate.freeSqliteEntries;
pub const DiagnosticReport = diagnostics.DiagnosticReport;
pub const CacheStats = diagnostics.CacheStats;
pub const diagnoseRuntime = diagnostics.diagnose;
pub const formatDiagnosticReport = diagnostics.formatReport;

// Extended retrieval stages
pub const expandQuery = query_expansion.expandQuery;
pub const ExpandedQuery = query_expansion.ExpandedQuery;
pub const analyzeQuery = adaptive.analyzeQuery;
pub const AdaptiveConfig = adaptive.AdaptiveConfig;
pub const QueryAnalysis = adaptive.QueryAnalysis;
pub const RetrievalStrategy = adaptive.RetrievalStrategy;
pub const buildRerankPrompt = llm_reranker.buildRerankPrompt;
pub const parseRerankResponse = llm_reranker.parseRerankResponse;
pub const LlmRerankerConfig = llm_reranker.LlmRerankerConfig;

// Lifecycle: summarizer
pub const SummarizerConfig = summarizer.SummarizerConfig;
pub const SummaryResult = summarizer.SummaryResult;
pub const shouldSummarize = summarizer.shouldSummarize;
pub const buildSummarizationPrompt = summarizer.buildSummarizationPrompt;
pub const parseSummaryResponse = summarizer.parseSummaryResponse;

// Lifecycle: semantic cache
pub const SemanticCache = semantic_cache.SemanticCache;

// ── Session message types ─────────────────────────────────────────

pub const MessageEntry = struct {
    role: []const u8,
    content: []const u8,
};

pub fn freeMessages(allocator: std.mem.Allocator, messages: []MessageEntry) void {
    for (messages) |entry| {
        allocator.free(entry.role);
        allocator.free(entry.content);
    }
    allocator.free(messages);
}

// ── SessionStore vtable interface ─────────────────────────────────

pub const SessionStore = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        saveMessage: *const fn (ptr: *anyopaque, session_id: []const u8, role: []const u8, content: []const u8) anyerror!void,
        loadMessages: *const fn (ptr: *anyopaque, allocator: std.mem.Allocator, session_id: []const u8) anyerror![]MessageEntry,
        clearMessages: *const fn (ptr: *anyopaque, session_id: []const u8) anyerror!void,
        clearAutoSaved: *const fn (ptr: *anyopaque, session_id: ?[]const u8) anyerror!void,
    };

    pub fn saveMessage(self: SessionStore, session_id: []const u8, role: []const u8, content: []const u8) !void {
        return self.vtable.saveMessage(self.ptr, session_id, role, content);
    }

    pub fn loadMessages(self: SessionStore, allocator: std.mem.Allocator, session_id: []const u8) ![]MessageEntry {
        return self.vtable.loadMessages(self.ptr, allocator, session_id);
    }

    pub fn clearMessages(self: SessionStore, session_id: []const u8) !void {
        return self.vtable.clearMessages(self.ptr, session_id);
    }

    pub fn clearAutoSaved(self: SessionStore, session_id: ?[]const u8) !void {
        return self.vtable.clearAutoSaved(self.ptr, session_id);
    }
};

// ── Memory categories ──────────────────────────────────────────────

pub const MemoryCategory = union(enum) {
    core,
    daily,
    conversation,
    custom: []const u8,

    pub fn toString(self: MemoryCategory) []const u8 {
        return switch (self) {
            .core => "core",
            .daily => "daily",
            .conversation => "conversation",
            .custom => |name| name,
        };
    }

    pub fn fromString(s: []const u8) MemoryCategory {
        if (std.mem.eql(u8, s, "core")) return .core;
        if (std.mem.eql(u8, s, "daily")) return .daily;
        if (std.mem.eql(u8, s, "conversation")) return .conversation;
        return .{ .custom = s };
    }

    pub fn eql(a: MemoryCategory, b: MemoryCategory) bool {
        const TagType = @typeInfo(MemoryCategory).@"union".tag_type.?;
        const tag_a: TagType = a;
        const tag_b: TagType = b;
        if (tag_a != tag_b) return false;
        if (tag_a == .custom) {
            return std.mem.eql(u8, a.custom, b.custom);
        }
        return true;
    }
};

// ── Memory entry ───────────────────────────────────────────────────

pub const MemoryEntry = struct {
    id: []const u8,
    key: []const u8,
    content: []const u8,
    category: MemoryCategory,
    timestamp: []const u8,
    session_id: ?[]const u8 = null,
    score: ?f64 = null,

    /// Free all allocated strings owned by this entry.
    pub fn deinit(self: *const MemoryEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.key);
        allocator.free(self.content);
        allocator.free(self.timestamp);
        if (self.session_id) |sid| allocator.free(sid);
        switch (self.category) {
            .custom => |name| allocator.free(name),
            else => {},
        }
    }
};

pub fn freeEntries(allocator: std.mem.Allocator, entries: []MemoryEntry) void {
    for (entries) |*entry| {
        entry.deinit(allocator);
    }
    allocator.free(entries);
}

// ── Memory vtable interface ────────────────────────────────────────

pub const Memory = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        name: *const fn (ptr: *anyopaque) []const u8,
        store: *const fn (ptr: *anyopaque, key: []const u8, content: []const u8, category: MemoryCategory, session_id: ?[]const u8) anyerror!void,
        recall: *const fn (ptr: *anyopaque, allocator: std.mem.Allocator, query: []const u8, limit: usize, session_id: ?[]const u8) anyerror![]MemoryEntry,
        get: *const fn (ptr: *anyopaque, allocator: std.mem.Allocator, key: []const u8) anyerror!?MemoryEntry,
        list: *const fn (ptr: *anyopaque, allocator: std.mem.Allocator, category: ?MemoryCategory, session_id: ?[]const u8) anyerror![]MemoryEntry,
        forget: *const fn (ptr: *anyopaque, key: []const u8) anyerror!bool,
        count: *const fn (ptr: *anyopaque) anyerror!usize,
        healthCheck: *const fn (ptr: *anyopaque) bool,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn name(self: Memory) []const u8 {
        return self.vtable.name(self.ptr);
    }

    pub fn store(self: Memory, key: []const u8, content: []const u8, category: MemoryCategory, session_id: ?[]const u8) !void {
        return self.vtable.store(self.ptr, key, content, category, session_id);
    }

    pub fn recall(self: Memory, allocator: std.mem.Allocator, query: []const u8, limit: usize, session_id: ?[]const u8) ![]MemoryEntry {
        return self.vtable.recall(self.ptr, allocator, query, limit, session_id);
    }

    pub fn get(self: Memory, allocator: std.mem.Allocator, key: []const u8) !?MemoryEntry {
        return self.vtable.get(self.ptr, allocator, key);
    }

    pub fn list(self: Memory, allocator: std.mem.Allocator, category: ?MemoryCategory, session_id: ?[]const u8) ![]MemoryEntry {
        return self.vtable.list(self.ptr, allocator, category, session_id);
    }

    pub fn forget(self: Memory, key: []const u8) !bool {
        return self.vtable.forget(self.ptr, key);
    }

    pub fn count(self: Memory) !usize {
        return self.vtable.count(self.ptr);
    }

    pub fn healthCheck(self: Memory) bool {
        return self.vtable.healthCheck(self.ptr);
    }

    pub fn deinit(self: Memory) void {
        self.vtable.deinit(self.ptr);
    }

    /// Hybrid search: combine keyword recall with optional vector similarity.
    /// This is a convenience method that wraps recall() and merges results.
    /// If an embedding provider is available, it can be used for vector search;
    /// otherwise falls back to keyword-only search via recall().
    pub fn search(self: Memory, allocator: std.mem.Allocator, query: []const u8, limit: usize) ![]MemoryEntry {
        // For now, delegate to recall() which uses FTS5/keyword search.
        // When embeddings are integrated at a higher level, this serves as
        // the standard entry point that can be upgraded to hybrid search.
        return self.recall(allocator, query, limit, null);
    }
};

// ── Backend kind classification ────────────────────────────────────

pub const MemoryBackendKind = enum {
    sqlite_backend,
    markdown_backend,
    none_backend,
    memory_lru_backend,
    lucid_backend,
    postgres_backend,
    redis_backend,
    lancedb_backend,
    unknown,
};

pub const MemoryBackendProfile = struct {
    key: []const u8,
    label: []const u8,
    auto_save_default: bool,
    uses_sqlite_hygiene: bool,
    sqlite_based: bool,
};

pub fn classifyBackend(backend_name: []const u8) MemoryBackendKind {
    if (std.mem.eql(u8, backend_name, "sqlite")) return .sqlite_backend;
    if (std.mem.eql(u8, backend_name, "markdown")) return .markdown_backend;
    if (std.mem.eql(u8, backend_name, "none")) return .none_backend;
    if (std.mem.eql(u8, backend_name, "memory")) return .memory_lru_backend;
    if (std.mem.eql(u8, backend_name, "lucid")) return .lucid_backend;
    if (std.mem.eql(u8, backend_name, "postgres")) return .postgres_backend;
    if (std.mem.eql(u8, backend_name, "redis")) return .redis_backend;
    if (std.mem.eql(u8, backend_name, "lancedb")) return .lancedb_backend;
    return .unknown;
}

pub fn defaultBackendKey() []const u8 {
    return "sqlite";
}

const base_selectable = [_]MemoryBackendProfile{
    .{
        .key = "sqlite",
        .label = "SQLite with FTS5 search (recommended)",
        .auto_save_default = true,
        .uses_sqlite_hygiene = true,
        .sqlite_based = true,
    },
    .{
        .key = "markdown",
        .label = "Markdown files — simple, human-readable",
        .auto_save_default = true,
        .uses_sqlite_hygiene = false,
        .sqlite_based = false,
    },
    .{
        .key = "lucid",
        .label = "Lucid — SQLite + cross-project memory sync via lucid CLI",
        .auto_save_default = true,
        .uses_sqlite_hygiene = true,
        .sqlite_based = true,
    },
    .{
        .key = "redis",
        .label = "Redis — distributed in-memory store with optional TTL",
        .auto_save_default = true,
        .uses_sqlite_hygiene = false,
        .sqlite_based = false,
    },
    .{
        .key = "lancedb",
        .label = "LanceDB — vector-augmented SQLite with duplicate detection",
        .auto_save_default = true,
        .uses_sqlite_hygiene = false,
        .sqlite_based = true,
    },
    .{
        .key = "none",
        .label = "None — disable persistent memory",
        .auto_save_default = false,
        .uses_sqlite_hygiene = false,
        .sqlite_based = false,
    },
};

const pg_selectable = if (build_options.enable_postgres) [_]MemoryBackendProfile{.{
    .key = "postgres",
    .label = "PostgreSQL — remote/shared memory store",
    .auto_save_default = true,
    .uses_sqlite_hygiene = false,
    .sqlite_based = false,
}} else [0]MemoryBackendProfile{};

pub const selectable_backends = base_selectable ++ pg_selectable;

// ── Factory ────────────────────────────────────────────────────────

pub const CreateError = error{
    SqliteOpenFailed,
    MigrationFailed,
    PrepareFailed,
    StepFailed,
    MarkdownInitFailed,
    InvalidBackend,
};

/// Create a memory backend by name. Caller owns the returned Memory and must call deinit().
/// For sqlite, pass the db_path (e.g. ":memory:" for tests, or a file path).
/// For markdown, pass workspace_dir as the path.
/// For none, path is ignored.
pub fn createMemory(allocator: std.mem.Allocator, backend_name: []const u8, path: [*:0]const u8) !Memory {
    const kind = classifyBackend(backend_name);
    return switch (kind) {
        .sqlite_backend => {
            const impl_ = try allocator.create(SqliteMemory);
            errdefer allocator.destroy(impl_);
            impl_.* = try SqliteMemory.init(allocator, path);
            impl_.owns_self = true;
            return impl_.memory();
        },
        .markdown_backend => {
            const impl_ = try allocator.create(MarkdownMemory);
            errdefer allocator.destroy(impl_);
            impl_.* = try MarkdownMemory.init(allocator, std.mem.span(path));
            impl_.owns_self = true;
            return impl_.memory();
        },
        .lucid_backend => {
            const impl_ = try allocator.create(LucidMemory);
            errdefer allocator.destroy(impl_);
            impl_.* = try LucidMemory.init(allocator, path, std.mem.span(path));
            impl_.owns_self = true;
            return impl_.memory();
        },
        .none_backend => {
            const impl_ = try allocator.create(NoneMemory);
            impl_.* = NoneMemory.init();
            impl_.allocator = allocator;
            return impl_.memory();
        },
        .memory_lru_backend => {
            const impl_ = try allocator.create(InMemoryLruMemory);
            impl_.* = InMemoryLruMemory.init(allocator, 1000);
            impl_.owns_self = true;
            return impl_.memory();
        },
        .lancedb_backend => {
            const impl_ = try allocator.create(LanceDbMemory);
            errdefer allocator.destroy(impl_);
            impl_.* = try LanceDbMemory.init(allocator, path, null, .{});
            impl_.owns_self = true;
            return impl_.memory();
        },
        .postgres_backend => return error.InvalidBackend, // postgres requires initRuntime with config
        .redis_backend => return error.InvalidBackend, // redis requires initRuntime with config
        .unknown => return error.InvalidBackend,
    };
}

// ── MemoryRuntime — bundled memory + session store + capabilities ──

pub const MemoryRuntime = struct {
    memory: Memory,
    session_store: ?SessionStore,
    response_cache: ?*cache.ResponseCache,
    capabilities: BackendCapabilities,

    // Internal: owned resources for cleanup
    _db_path: ?[*:0]const u8,
    _cache_db_path: ?[*:0]const u8,
    _engine: ?*retrieval.RetrievalEngine,
    _allocator: std.mem.Allocator,

    // P5: rollout policy
    _rollout_policy: rollout.RolloutPolicy = .{ .mode = .on, .canary_percent = 0, .shadow_percent = 0 },

    // Lifecycle: summarizer config
    _summarizer_cfg: summarizer.SummarizerConfig = .{},

    // Lifecycle: semantic cache (optional, extends response cache with cosine similarity)
    _semantic_cache: ?*semantic_cache.SemanticCache = null,

    // P3: vector plane components (all optional)
    _embedding_provider: ?embeddings.EmbeddingProvider = null,
    _vector_store_impl: ?*vector_store.SqliteSharedVectorStore = null,
    _circuit_breaker: ?*circuit_breaker.CircuitBreaker = null,
    _outbox: ?*outbox.VectorOutbox = null,

    /// High-level search: uses rollout policy to decide keyword-only vs hybrid.
    pub fn search(self: *MemoryRuntime, allocator: std.mem.Allocator, query: []const u8, limit: usize, session_id: ?[]const u8) ![]RetrievalCandidate {
        const decision = self._rollout_policy.decide(session_id);

        switch (decision) {
            .keyword_only => {
                // Bypass engine, use recall() directly
                const entries = try self.memory.recall(allocator, query, limit, session_id);
                defer freeEntries(allocator, entries);
                return retrieval.entriesToCandidates(allocator, entries);
            },
            .hybrid => {
                // Use engine if available, else fall back
                if (self._engine) |engine| {
                    return engine.search(allocator, query, session_id);
                }
                const entries = try self.memory.recall(allocator, query, limit, session_id);
                defer freeEntries(allocator, entries);
                return retrieval.entriesToCandidates(allocator, entries);
            },
            .shadow_hybrid => {
                // Run both, serve keyword result, log hybrid for comparison
                const keyword_entries = try self.memory.recall(allocator, query, limit, session_id);
                defer freeEntries(allocator, keyword_entries);
                const keyword_results = try retrieval.entriesToCandidates(allocator, keyword_entries);

                if (self._engine) |engine| {
                    const hybrid_results = engine.search(allocator, query, session_id) catch |err| {
                        log.warn("shadow hybrid search failed: {}", .{err});
                        return keyword_results;
                    };
                    defer retrieval.freeCandidates(allocator, hybrid_results);

                    log.info("shadow: keyword={d} hybrid={d} results", .{ keyword_results.len, hybrid_results.len });
                }

                return keyword_results;
            },
        }
    }

    /// Get current rollout mode.
    pub fn rolloutMode(self: *const MemoryRuntime) rollout.RolloutMode {
        return self._rollout_policy.mode;
    }

    /// Best-effort vector sync after a store() call.
    /// Embeds the content and upserts into the vector store.
    /// Errors are caught and logged, never propagated.
    pub fn syncVectorAfterStore(self: *MemoryRuntime, allocator: std.mem.Allocator, key: []const u8, content: []const u8) void {
        const provider = self._embedding_provider orelse return;
        const vs_impl = self._vector_store_impl orelse return;
        const vs = vs_impl.store();

        // Check circuit breaker
        if (self._circuit_breaker) |cb| {
            if (!cb.allow()) return;
        }

        const emb = provider.embed(allocator, content) catch |err| {
            log.warn("vector sync embed failed for key '{s}': {}", .{ key, err });
            if (self._circuit_breaker) |cb| cb.recordFailure();
            return;
        };
        defer allocator.free(emb);

        if (self._circuit_breaker) |cb| cb.recordSuccess();
        if (emb.len == 0) return;

        vs.upsert(key, emb) catch |err| {
            log.warn("vector sync upsert failed for key '{s}': {}", .{ key, err });
        };
    }

    /// Drain the durable outbox (if configured).
    /// Call periodically (e.g., after each agent turn).
    pub fn drainOutbox(self: *MemoryRuntime, allocator: std.mem.Allocator) u32 {
        const ob = self._outbox orelse return 0;
        const provider = self._embedding_provider orelse return 0;
        const vs_impl = self._vector_store_impl orelse return 0;
        const vs = vs_impl.store();
        return ob.drain(allocator, provider, vs, self._circuit_breaker) catch 0;
    }

    /// Enqueue a key for vector sync via the outbox (if configured).
    pub fn enqueueVectorSync(self: *MemoryRuntime, key: []const u8, operation: []const u8) void {
        const ob = self._outbox orelse return;
        ob.enqueue(key, operation) catch |err| {
            log.warn("outbox enqueue failed for key '{s}': {}", .{ key, err });
        };
    }

    /// Get the summarizer configuration (for the agent/session layer to use).
    pub fn summarizerConfig(self: *const MemoryRuntime) summarizer.SummarizerConfig {
        return self._summarizer_cfg;
    }

    /// Get the semantic cache (for the agent/session layer to use).
    pub fn semanticCache(self: *MemoryRuntime) ?*semantic_cache.SemanticCache {
        return self._semantic_cache;
    }

    /// Run memory doctor diagnostics and return a report.
    pub fn diagnose(self: *MemoryRuntime) diagnostics.DiagnosticReport {
        return diagnostics.diagnose(self);
    }

    pub fn deinit(self: *MemoryRuntime) void {
        // P3 cleanup (before engine, since engine may reference vector store)
        if (self._outbox) |ob| {
            ob.deinit();
            self._allocator.destroy(ob);
        }
        if (self._circuit_breaker) |cb| {
            self._allocator.destroy(cb);
        }
        if (self._vector_store_impl) |vs| {
            vs.deinit();
            self._allocator.destroy(vs);
        }
        if (self._embedding_provider) |ep| {
            ep.deinit();
        }

        if (self._engine) |engine| {
            engine.deinit();
            self._allocator.destroy(engine);
        }
        if (self._semantic_cache) |sc| {
            sc.deinit();
            self._allocator.destroy(sc);
        }
        if (self.response_cache) |rc| {
            rc.deinit();
            self._allocator.destroy(rc);
        }
        if (self._cache_db_path) |p| self._allocator.free(std.mem.span(p));
        self.memory.deinit();
        if (self._db_path) |p| self._allocator.free(std.mem.span(p));
    }
};

/// Create a MemoryRuntime from a MemoryConfig and workspace directory.
/// Goes through the registry to find the backend, resolve paths, and
/// create the instance. Returns null on any error (unknown backend,
/// path resolution failure, backend init failure).
pub fn initRuntime(
    allocator: std.mem.Allocator,
    config: *const config_types.MemoryConfig,
    workspace_dir: []const u8,
) ?MemoryRuntime {
    const desc = registry.findBackend(config.backend) orelse return null;

    const pg_cfg: ?config_types.MemoryPostgresConfig = if (std.mem.eql(u8, config.backend, "postgres")) config.postgres else null;
    const cfg = registry.resolvePaths(allocator, desc, workspace_dir, pg_cfg) catch return null;

    const instance = desc.create(allocator, cfg) catch {
        if (cfg.postgres_url) |pu| allocator.free(std.mem.span(pu));
        if (cfg.db_path) |p| allocator.free(std.mem.span(p));
        return null;
    };

    // ── Lifecycle: snapshot hydrate (before hygiene) ──
    if (config.lifecycle.auto_hydrate) {
        if (snapshot.shouldHydrate(allocator, instance.memory, workspace_dir)) {
            _ = snapshot.hydrateFromSnapshot(allocator, instance.memory, workspace_dir) catch |e| {
                log.warn("snapshot hydration failed: {}", .{e});
            };
        }
    }

    // ── Lifecycle: hygiene ──
    if (config.lifecycle.hygiene_enabled) {
        const hygiene_cfg = hygiene.HygieneConfig{
            .hygiene_enabled = true,
            .archive_after_days = config.lifecycle.archive_after_days,
            .purge_after_days = config.lifecycle.purge_after_days,
            .conversation_retention_days = config.lifecycle.conversation_retention_days,
            .workspace_dir = workspace_dir,
        };
        const report = hygiene.runIfDue(allocator, hygiene_cfg, instance.memory);

        // Snapshot after hygiene if configured and hygiene did work
        if (config.lifecycle.snapshot_on_hygiene and report.totalActions() > 0) {
            _ = snapshot.exportSnapshot(allocator, instance.memory, workspace_dir) catch |e| {
                log.warn("snapshot export after hygiene failed: {}", .{e});
            };
        }
    }

    // ── Lifecycle: response cache ──
    var resp_cache: ?*cache.ResponseCache = null;
    var cache_db_path: ?[*:0]const u8 = null;
    if (config.response_cache.enabled) blk: {
        const cp_slice = std.fs.path.joinZ(allocator, &.{ workspace_dir, "response_cache.db" }) catch break :blk;
        const cp: [*:0]const u8 = cp_slice.ptr;
        const rc = allocator.create(cache.ResponseCache) catch {
            allocator.free(std.mem.span(cp));
            break :blk;
        };
        rc.* = cache.ResponseCache.init(cp, config.response_cache.ttl_minutes, config.response_cache.max_entries) catch {
            allocator.destroy(rc);
            allocator.free(std.mem.span(cp));
            break :blk;
        };
        resp_cache = rc;
        cache_db_path = cp;
    }

    // ── Retrieval engine ──
    var engine: ?*retrieval.RetrievalEngine = null;
    build_engine: {
        const eng = allocator.create(retrieval.RetrievalEngine) catch break :build_engine;
        eng.* = retrieval.RetrievalEngine.init(allocator, config.search.query);

        // Always add primary adapter
        const primary = allocator.create(retrieval.PrimaryAdapter) catch {
            allocator.destroy(eng);
            break :build_engine;
        };
        primary.* = retrieval.PrimaryAdapter.init(instance.memory);
        primary.owns_self = true;
        primary.allocator = allocator;
        eng.addSource(primary.adapter()) catch {
            allocator.destroy(primary);
            eng.deinit();
            allocator.destroy(eng);
            break :build_engine;
        };

        // QMD adapter (optional)
        if (config.qmd.enabled) {
            const qmd = allocator.create(retrieval_qmd.QmdAdapter) catch break :build_engine;
            qmd.* = retrieval_qmd.QmdAdapter.init(allocator, config.qmd, workspace_dir);
            qmd.owns_self = true;
            eng.addSource(qmd.adapter()) catch {
                allocator.destroy(qmd);
                // engine still usable without QMD
            };
        }

        // Configure extended pipeline stages (query expansion, adaptive, LLM reranker)
        eng.setRetrievalStages(config.retrieval_stages);

        engine = eng;
    }

    // ── P3: Vector plane wiring ──
    var embed_provider: ?embeddings.EmbeddingProvider = null;
    var vs_impl: ?*vector_store.SqliteSharedVectorStore = null;
    var cb_inst: ?*circuit_breaker.CircuitBreaker = null;
    var outbox_inst: ?*outbox.VectorOutbox = null;

    if (!std.mem.eql(u8, config.search.provider, "none") and config.search.query.hybrid.enabled) vec_plane: {
        // 1. Create EmbeddingProvider
        const ep = embeddings.createEmbeddingProvider(
            allocator,
            config.search.provider,
            null,
            config.search.model,
            config.search.dimensions,
        ) catch break :vec_plane;
        embed_provider = ep;

        // 2. Extract sqlite db handle
        const db_handle = extractSqliteDb(instance.memory) orelse break :vec_plane;

        // 3. Create SqliteSharedVectorStore
        const vs = allocator.create(vector_store.SqliteSharedVectorStore) catch break :vec_plane;
        vs.* = vector_store.SqliteSharedVectorStore.init(allocator, db_handle);
        vs.owns_self = true;
        vs_impl = vs;

        // 4. Create CircuitBreaker
        const cb = allocator.create(circuit_breaker.CircuitBreaker) catch break :vec_plane;
        cb.* = circuit_breaker.CircuitBreaker.init(
            config.reliability.circuit_breaker_failures,
            config.reliability.circuit_breaker_cooldown_ms,
        );
        cb_inst = cb;

        // 5. Create VectorOutbox if not best_effort
        if (!std.mem.eql(u8, config.search.sync.mode, "best_effort")) {
            const ob = allocator.create(outbox.VectorOutbox) catch break :vec_plane;
            ob.* = outbox.VectorOutbox.init(allocator, db_handle, config.search.sync.embed_max_retries);
            ob.owns_self = true;
            ob.migrate() catch {
                allocator.destroy(ob);
                break :vec_plane;
            };
            outbox_inst = ob;
        }

        // 6. Wire into retrieval engine
        if (engine) |eng| {
            const vs_iface = vs.store();
            eng.setVectorSearch(ep, vs_iface, cb, config.search.query.hybrid);
        }
    }

    // Free postgres_url after backend creation (backend dupes what it needs)
    if (cfg.postgres_url) |pu| allocator.free(std.mem.span(pu));

    // ── Lifecycle: semantic cache ──
    var sem_cache: ?*semantic_cache.SemanticCache = null;
    if (config.response_cache.enabled and embed_provider != null) sem_cache_blk: {
        const sc_path = std.fs.path.joinZ(allocator, &.{ workspace_dir, "semantic_cache.db" }) catch break :sem_cache_blk;
        const sc = allocator.create(semantic_cache.SemanticCache) catch {
            allocator.free(std.mem.span(sc_path.ptr));
            break :sem_cache_blk;
        };
        sc.* = semantic_cache.SemanticCache.init(
            sc_path.ptr,
            config.response_cache.ttl_minutes,
            config.response_cache.max_entries,
            0.95, // cosine similarity threshold
            embed_provider,
        ) catch {
            allocator.destroy(sc);
            allocator.free(std.mem.span(sc_path.ptr));
            break :sem_cache_blk;
        };
        sem_cache = sc;
        // Note: sc_path is owned by the semantic cache's sqlite connection
    }

    // ── Lifecycle: summarizer config ──
    const summarizer_cfg = summarizer.SummarizerConfig{
        .enabled = config.summarizer.enabled,
        .window_size_tokens = @intCast(config.summarizer.window_size_tokens),
        .summary_max_tokens = @intCast(config.summarizer.summary_max_tokens),
        .auto_extract_semantic = config.summarizer.auto_extract_semantic,
    };

    // ── Startup diagnostic ──
    const source_count: usize = if (engine) |eng| eng.sources.items.len else 0;
    const vector_mode: []const u8 = if (vs_impl != null) "sqlite_shared" else "none";
    log.info("memory plan resolved: backend={s} retrieval={s} vector={s} rollout={s} hygiene={} snapshot={} cache={} semantic_cache={} summarizer={} sources={d}", .{
        config.backend,
        if (config.search.query.hybrid.enabled) "hybrid" else "keyword",
        vector_mode,
        config.reliability.rollout_mode,
        config.lifecycle.hygiene_enabled,
        config.lifecycle.snapshot_enabled,
        config.response_cache.enabled,
        sem_cache != null,
        config.summarizer.enabled,
        source_count,
    });

    return .{
        .memory = instance.memory,
        .session_store = instance.session_store,
        .response_cache = resp_cache,
        .capabilities = desc.capabilities,
        ._db_path = cfg.db_path,
        ._cache_db_path = cache_db_path,
        ._engine = engine,
        ._allocator = allocator,
        ._rollout_policy = rollout.RolloutPolicy.init(config.reliability),
        ._summarizer_cfg = summarizer_cfg,
        ._semantic_cache = sem_cache,
        ._embedding_provider = embed_provider,
        ._vector_store_impl = vs_impl,
        ._circuit_breaker = cb_inst,
        ._outbox = outbox_inst,
    };
}

// ── Helpers ────────────────────────────────────────────────────────

const c = sqlite.c;

/// Extract the raw sqlite3* handle from a Memory vtable, if the backend is sqlite-based.
fn extractSqliteDb(mem: Memory) ?*c.sqlite3 {
    const name_str = mem.name();
    if (std.mem.eql(u8, name_str, "sqlite")) {
        const impl_: *SqliteMemory = @ptrCast(@alignCast(mem.ptr));
        return impl_.db;
    }
    if (std.mem.eql(u8, name_str, "lucid")) {
        const impl_: *LucidMemory = @ptrCast(@alignCast(mem.ptr));
        return impl_.local.db;
    }
    return null;
}

// ── Tests ──────────────────────────────────────────────────────────

test "MemoryCategory toString roundtrip" {
    const core: MemoryCategory = .core;
    try std.testing.expectEqualStrings("core", core.toString());

    const daily: MemoryCategory = .daily;
    try std.testing.expectEqualStrings("daily", daily.toString());

    const conversation: MemoryCategory = .conversation;
    try std.testing.expectEqualStrings("conversation", conversation.toString());

    const custom: MemoryCategory = .{ .custom = "project" };
    try std.testing.expectEqualStrings("project", custom.toString());
}

test "MemoryCategory fromString" {
    const core = MemoryCategory.fromString("core");
    try std.testing.expect(core.eql(.core));

    const daily = MemoryCategory.fromString("daily");
    try std.testing.expect(daily.eql(.daily));

    const conversation = MemoryCategory.fromString("conversation");
    try std.testing.expect(conversation.eql(.conversation));

    const custom = MemoryCategory.fromString("project");
    try std.testing.expectEqualStrings("project", custom.custom);
}

test "MemoryCategory equality" {
    const core: MemoryCategory = .core;
    try std.testing.expect(core.eql(.core));
    try std.testing.expect(!core.eql(.daily));
    const c1: MemoryCategory = .{ .custom = "a" };
    const c2: MemoryCategory = .{ .custom = "a" };
    const c3: MemoryCategory = .{ .custom = "b" };
    try std.testing.expect(c1.eql(c2));
    try std.testing.expect(!c1.eql(c3));
}

test "classifyBackend" {
    try std.testing.expect(classifyBackend("sqlite") == .sqlite_backend);
    try std.testing.expect(classifyBackend("markdown") == .markdown_backend);
    try std.testing.expect(classifyBackend("none") == .none_backend);
    try std.testing.expect(classifyBackend("lucid") == .lucid_backend);
    try std.testing.expect(classifyBackend("postgres") == .postgres_backend);
    try std.testing.expect(classifyBackend("redis") == .redis_backend);
    try std.testing.expect(classifyBackend("lancedb") == .lancedb_backend);
}

test "selectable backends are ordered" {
    const expected: usize = if (build_options.enable_postgres) 7 else 6;
    try std.testing.expect(selectable_backends.len == expected);
    try std.testing.expectEqualStrings("sqlite", selectable_backends[0].key);
    try std.testing.expectEqualStrings("markdown", selectable_backends[1].key);
    try std.testing.expectEqualStrings("lucid", selectable_backends[2].key);
    try std.testing.expectEqualStrings("redis", selectable_backends[3].key);
    try std.testing.expectEqualStrings("lancedb", selectable_backends[4].key);
    try std.testing.expectEqualStrings("none", selectable_backends[5].key);
}

test "defaultBackendKey is sqlite" {
    try std.testing.expectEqualStrings("sqlite", defaultBackendKey());
}

test "MemoryCategory custom toString" {
    const cat: MemoryCategory = .{ .custom = "my_project" };
    try std.testing.expectEqualStrings("my_project", cat.toString());
}

test "MemoryCategory fromString custom" {
    const cat = MemoryCategory.fromString("unknown_category");
    try std.testing.expectEqualStrings("unknown_category", cat.custom);
}

test "MemoryCategory eql different tags" {
    const core: MemoryCategory = .core;
    const daily: MemoryCategory = .daily;
    const conv: MemoryCategory = .conversation;
    try std.testing.expect(!core.eql(daily));
    try std.testing.expect(!core.eql(conv));
    try std.testing.expect(!daily.eql(conv));
}

test "classifyBackend unknown returns unknown" {
    try std.testing.expect(classifyBackend("nonexistent") == .unknown);
    try std.testing.expect(classifyBackend("") == .unknown);
    try std.testing.expect(classifyBackend("SQLITE") == .unknown);
}

test "createMemory unknown backend returns InvalidBackend" {
    const result = createMemory(std.testing.allocator, "nonexistent", "unused");
    try std.testing.expectError(error.InvalidBackend, result);
}

test "createMemory redis backend returns InvalidBackend" {
    const result = createMemory(std.testing.allocator, "redis", "unused");
    try std.testing.expectError(error.InvalidBackend, result);
}

test "selectable backends sqlite is recommended" {
    try std.testing.expect(selectable_backends[0].sqlite_based);
    try std.testing.expect(selectable_backends[0].uses_sqlite_hygiene);
    try std.testing.expect(selectable_backends[0].auto_save_default);
}

test "selectable backends lucid is sqlite based" {
    try std.testing.expect(selectable_backends[2].auto_save_default);
    try std.testing.expect(selectable_backends[2].sqlite_based);
    try std.testing.expect(selectable_backends[2].uses_sqlite_hygiene);
}

test "selectable backends redis is not sqlite based" {
    try std.testing.expect(selectable_backends[3].auto_save_default);
    try std.testing.expect(!selectable_backends[3].sqlite_based);
    try std.testing.expect(!selectable_backends[3].uses_sqlite_hygiene);
}

test "selectable backends lancedb is sqlite based" {
    try std.testing.expect(selectable_backends[4].auto_save_default);
    try std.testing.expect(selectable_backends[4].sqlite_based);
    try std.testing.expect(!selectable_backends[4].uses_sqlite_hygiene);
}

test "selectable backends none has no auto save" {
    try std.testing.expect(!selectable_backends[5].auto_save_default);
    try std.testing.expect(!selectable_backends[5].sqlite_based);
    try std.testing.expect(!selectable_backends[5].uses_sqlite_hygiene);
}

test "Memory convenience store accepts session_id" {
    var backend = none.NoneMemory.init();
    defer backend.deinit();
    const m = backend.memory();
    try m.store("key", "value", .core, null);
    try m.store("key2", "value2", .daily, "session-abc");
}

test "Memory convenience recall accepts session_id" {
    var backend = none.NoneMemory.init();
    defer backend.deinit();
    const m = backend.memory();
    const results = try m.recall(std.testing.allocator, "query", 5, null);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);

    const results2 = try m.recall(std.testing.allocator, "query", 5, "session-abc");
    defer std.testing.allocator.free(results2);
    try std.testing.expectEqual(@as(usize, 0), results2.len);
}

test "Memory convenience list accepts session_id" {
    var backend = none.NoneMemory.init();
    defer backend.deinit();
    const m = backend.memory();
    const results = try m.list(std.testing.allocator, null, null);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);

    const results2 = try m.list(std.testing.allocator, .core, "session-abc");
    defer std.testing.allocator.free(results2);
    try std.testing.expectEqual(@as(usize, 0), results2.len);
}

test "SessionStore delegates through vtable" {
    const TestSessionStore = struct {
        call_count: usize = 0,

        fn implSaveMessage(ptr: *anyopaque, _: []const u8, _: []const u8, _: []const u8) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.call_count += 1;
        }
        fn implLoadMessages(_: *anyopaque, allocator: std.mem.Allocator, _: []const u8) anyerror![]MessageEntry {
            return allocator.alloc(MessageEntry, 0);
        }
        fn implClearMessages(ptr: *anyopaque, _: []const u8) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.call_count += 1;
        }
        fn implClearAutoSaved(ptr: *anyopaque, _: ?[]const u8) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.call_count += 1;
        }

        const sess_vtable = SessionStore.VTable{
            .saveMessage = &implSaveMessage,
            .loadMessages = &implLoadMessages,
            .clearMessages = &implClearMessages,
            .clearAutoSaved = &implClearAutoSaved,
        };
    };

    var mock = TestSessionStore{};
    const store = SessionStore{ .ptr = @ptrCast(&mock), .vtable = &TestSessionStore.sess_vtable };

    try store.saveMessage("s1", "user", "hello");
    try std.testing.expectEqual(@as(usize, 1), mock.call_count);

    const msgs = try store.loadMessages(std.testing.allocator, "s1");
    defer std.testing.allocator.free(msgs);
    try std.testing.expectEqual(@as(usize, 0), msgs.len);

    try store.clearMessages("s1");
    try std.testing.expectEqual(@as(usize, 2), mock.call_count);

    try store.clearAutoSaved(null);
    try std.testing.expectEqual(@as(usize, 3), mock.call_count);
}

test "freeMessages frees all entries" {
    const allocator = std.testing.allocator;
    var messages = try allocator.alloc(MessageEntry, 2);
    messages[0] = .{ .role = try allocator.dupe(u8, "user"), .content = try allocator.dupe(u8, "hello") };
    messages[1] = .{ .role = try allocator.dupe(u8, "assistant"), .content = try allocator.dupe(u8, "hi") };
    freeMessages(allocator, messages);
    // No leak = pass (allocator is testing allocator with leak detection)
}

test "initRuntime none returns valid runtime" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp") orelse
        return error.TestUnexpectedResult;
    defer rt.deinit();

    try std.testing.expectEqualStrings("none", rt.memory.name());
    try std.testing.expect(rt.session_store == null);
    try std.testing.expect(!rt.capabilities.supports_session_store);
    try std.testing.expect(!rt.capabilities.supports_keyword_rank);
}

test "initRuntime unknown backend returns null" {
    try std.testing.expect(initRuntime(std.testing.allocator, &.{ .backend = "unknown_backend" }, "/tmp") == null);
}

test "initRuntime none deinit does not leak" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp") orelse
        return error.TestUnexpectedResult;
    rt.deinit();
    // testing allocator detects leaks — if we get here, no leak
}

test "initRuntime none has null db_path" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp") orelse
        return error.TestUnexpectedResult;
    defer rt.deinit();

    try std.testing.expect(rt._db_path == null);
    try std.testing.expect(rt.response_cache == null);
}

test "initRuntime sqlite returns full runtime" {
    var rt = initRuntime(std.testing.allocator, &.{}, "/tmp") orelse
        return error.TestUnexpectedResult;
    defer rt.deinit();

    try std.testing.expectEqualStrings("sqlite", rt.memory.name());
    try std.testing.expect(rt.session_store != null);
    try std.testing.expect(rt.capabilities.supports_session_store);
    try std.testing.expect(rt.capabilities.supports_keyword_rank);
    try std.testing.expect(rt.capabilities.supports_transactions);
    try std.testing.expect(rt._db_path != null);
    const path_slice = std.mem.span(rt._db_path.?);
    try std.testing.expect(std.mem.endsWith(u8, path_slice, "memory.db"));
}

test "initRuntime with lifecycle defaults does not crash" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp/test_lifecycle");
    if (rt) |*r| r.deinit();
}

test "initRuntime with cache disabled leaves response_cache null" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp/test_nocache") orelse return;
    defer rt.deinit();
    try std.testing.expect(rt.response_cache == null);
    try std.testing.expect(rt._cache_db_path == null);
}

test "initRuntime with cache enabled creates ResponseCache" {
    var rt = initRuntime(std.testing.allocator, &.{
        .backend = "none",
        .response_cache = .{
            .enabled = true,
            .ttl_minutes = 5,
            .max_entries = 100,
        },
    }, "/tmp") orelse return;
    defer rt.deinit();
    try std.testing.expect(rt.response_cache != null);
    try std.testing.expect(rt._cache_db_path != null);
}

test "initRuntime creates engine with primary source" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp") orelse
        return error.TestUnexpectedResult;
    defer rt.deinit();
    try std.testing.expect(rt._engine != null);
}

test "initRuntime engine with qmd disabled has one source" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp") orelse
        return error.TestUnexpectedResult;
    defer rt.deinit();
    if (rt._engine) |eng| {
        try std.testing.expectEqual(@as(usize, 1), eng.sources.items.len);
    }
}

test "MemoryRuntime.search without engine falls back to recall" {
    var backend = none.NoneMemory.init();
    defer backend.deinit();
    var rt = MemoryRuntime{
        .memory = backend.memory(),
        .session_store = null,
        .response_cache = null,
        .capabilities = .{ .supports_keyword_rank = false, .supports_session_store = false, .supports_transactions = false, .supports_outbox = false },
        ._db_path = null,
        ._cache_db_path = null,
        ._engine = null,
        ._allocator = std.testing.allocator,
        ._embedding_provider = null,
        ._vector_store_impl = null,
        ._circuit_breaker = null,
        ._outbox = null,
    };
    const results = try rt.search(std.testing.allocator, "query", 5, null);
    defer retrieval.freeCandidates(std.testing.allocator, results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "MemoryRuntime.search with engine delegates" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp") orelse
        return error.TestUnexpectedResult;
    defer rt.deinit();
    const results = try rt.search(std.testing.allocator, "query", 5, null);
    defer retrieval.freeCandidates(std.testing.allocator, results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "initRuntime with hybrid disabled has no embedding provider" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp") orelse
        return error.TestUnexpectedResult;
    defer rt.deinit();

    try std.testing.expect(rt._embedding_provider == null);
    try std.testing.expect(rt._vector_store_impl == null);
    try std.testing.expect(rt._circuit_breaker == null);
    try std.testing.expect(rt._outbox == null);
}

test "initRuntime with search.provider=none has no vector store" {
    var rt = initRuntime(std.testing.allocator, &.{
        .backend = "none",
        .search = .{
            .provider = "none",
            .query = .{ .hybrid = .{ .enabled = true } },
        },
    }, "/tmp") orelse
        return error.TestUnexpectedResult;
    defer rt.deinit();

    try std.testing.expect(rt._embedding_provider == null);
    try std.testing.expect(rt._vector_store_impl == null);
}

test "MemoryRuntime.syncVectorAfterStore with no provider is no-op" {
    var backend = none.NoneMemory.init();
    defer backend.deinit();
    var rt = MemoryRuntime{
        .memory = backend.memory(),
        .session_store = null,
        .response_cache = null,
        .capabilities = .{ .supports_keyword_rank = false, .supports_session_store = false, .supports_transactions = false, .supports_outbox = false },
        ._db_path = null,
        ._cache_db_path = null,
        ._engine = null,
        ._allocator = std.testing.allocator,
        ._embedding_provider = null,
        ._vector_store_impl = null,
        ._circuit_breaker = null,
        ._outbox = null,
    };
    // Should not crash — just a no-op
    rt.syncVectorAfterStore(std.testing.allocator, "key", "content");
}

test "MemoryRuntime.drainOutbox with no outbox returns 0" {
    var backend = none.NoneMemory.init();
    defer backend.deinit();
    var rt = MemoryRuntime{
        .memory = backend.memory(),
        .session_store = null,
        .response_cache = null,
        .capabilities = .{ .supports_keyword_rank = false, .supports_session_store = false, .supports_transactions = false, .supports_outbox = false },
        ._db_path = null,
        ._cache_db_path = null,
        ._engine = null,
        ._allocator = std.testing.allocator,
        ._embedding_provider = null,
        ._vector_store_impl = null,
        ._circuit_breaker = null,
        ._outbox = null,
    };
    try std.testing.expectEqual(@as(u32, 0), rt.drainOutbox(std.testing.allocator));
}

test "MemoryRuntime.deinit cleans up P3 resources" {
    var rt = initRuntime(std.testing.allocator, &.{ .backend = "none" }, "/tmp") orelse
        return error.TestUnexpectedResult;
    // P3 fields are null for "none" backend with hybrid disabled, but deinit should handle that.
    rt.deinit();
    // testing allocator detects leaks
}

test {
    // engines/ (Layer A)
    _ = sqlite;
    _ = markdown;
    _ = none;
    _ = memory_lru;
    _ = lucid;
    _ = postgres;
    _ = redis;
    _ = lancedb;
    _ = registry;
    _ = @import("engines/contract_test.zig");

    // retrieval/ (Layer B)
    _ = retrieval;
    _ = retrieval_qmd;
    _ = rrf;
    _ = query_expansion;
    _ = temporal_decay;
    _ = mmr;
    _ = adaptive;
    _ = llm_reranker;

    // vector/ (Layer C)
    _ = vector;
    _ = vector_store;
    _ = embeddings;
    _ = embeddings_gemini;
    _ = embeddings_voyage;
    _ = embeddings_ollama;
    _ = provider_router;
    _ = store_qdrant;
    _ = store_pgvector;
    _ = circuit_breaker;
    _ = outbox;
    _ = chunker;

    // lifecycle/ (Layer D)
    _ = cache;
    _ = semantic_cache;
    _ = hygiene;
    _ = snapshot;
    _ = rollout;
    _ = migrate;
    _ = diagnostics;
    _ = summarizer;
}
