/* =========================================================================
 * KV Cache.
 * =========================================================================
 *
 * The server has one live Metal session.  We persist reusable DS4 session
 * snapshots when a cold prompt reaches a useful prefix, when a long continued
 * conversation has grown far enough, and when a request evicts the live session.
 * The cache key is the SHA1 of the rendered byte prefix.  The payload still
 * stores exact token IDs and graph state; the filename only selects a checkpoint
 * whose decoded transcript bytes are a prefix of the next rendered request.
 *
 * Files are loaded with plain read/write I/O into the existing graph tensors;
 * mmap is deliberately avoided here so cache restore cannot add more VM
 * mappings to a process that already maps a very large GGUF.
 *
 * Stores are created only when the live graph is already at the checkpoint we
 * want to persist.  For long cold prompts this means prefill reaches the stable
 * boundary first, writes that prefix, and then continues with the suffix.  We
 * never roll the session backward just to build a disk cache entry: that would
 * turn cache population into a second hidden prefill.
 *
 * File layout:
 *
 *   "KVC" version
 *   quant bits, save reason, token count, hit count, context size
 *   creation time, last-used time, payload byte count
 *   rendered text byte count + rendered text for human inspection
 *   DS4 engine payload written by ds4_session_save_payload()
 *   optional tool-id map section
 *
 * The filename is SHA1(cache text bytes), not SHA1(token ids).  For ordinary
 * checkpoints the cache text is the rendered token prefix.  For live hidden
 * state it can instead be the client-visible transcript: the payload still
 * contains sampled reasoning KV, but the lookup key must be what the client can
 * replay after a process restart or session switch.
 *
 * The optional tool-id map is not part of model state, but it is needed to
 * render future client JSON back to the exact DSML sampled by the model.  We
 * persist only mappings whose DSML block appears in the saved cache text.
 */

#define KV_CACHE_MAGIC0 'K'
#define KV_CACHE_MAGIC1 'V'
#define KV_CACHE_MAGIC2 'C'
#define KV_CACHE_VERSION 1u
#define KV_CACHE_FIXED_HEADER 48u
#define KV_CACHE_DEFAULT_MIN_TOKENS 512
#define KV_CACHE_DEFAULT_COLD_MAX_TOKENS 30000
/* Tokenizers may merge text across the prompt boundary.  Trimming a small tail
 * still improves the cheap token-prefix path, while text-prefix lookup handles
 * the cases where canonical prompt tokenization spells the same bytes
 * differently.  The 2048 alignment also matches the Metal prefill chunk
 * schedule, which keeps compressor row finalization identical to a cold full
 * prompt. */
#define KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS 32
#define KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS 2048
#define KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS 10000
#define KV_CACHE_DEFAULT_MB 4096
/* Disk-hit counts are evidence that a checkpoint was useful, but only while
 * the workload still resembles the one that produced those hits.  Prompt or
 * tool-schema changes can make a once-hot checkpoint impossible to match, so
 * eviction decays only the hit bonus with inactivity.  The baseline token/byte
 * value remains intact: old files are not punished for age, they just stop
 * carrying stale popularity forever. */
#define KV_CACHE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)
#define KV_CACHE_MIN_EFFECTIVE_HITS 0.01
#define KV_EXT_TOOL_MAP (1u << 0)
#define KV_EXT_RESPONSES_VISIBLE (1u << 1)
#define KV_EXT_THINKING_VISIBLE (1u << 2)
#define KV_TOOL_MAP_MAGIC0 'K'
#define KV_TOOL_MAP_MAGIC1 'T'
#define KV_TOOL_MAP_MAGIC2 'M'
#define KV_TOOL_MAP_VERSION 1u
#define KV_TOOL_MAP_HEADER 8u

typedef enum {
    KV_REASON_UNKNOWN   = 0,
    KV_REASON_COLD      = 1,
    KV_REASON_CONTINUED = 2,
    KV_REASON_EVICT     = 3,
    KV_REASON_SHUTDOWN  = 4,
} kv_cache_reason;

static uint8_t kv_reason_code(const char *reason) {
    if (!reason) return KV_REASON_UNKNOWN;
    if (!strcmp(reason, "cold")) return KV_REASON_COLD;
    if (!strcmp(reason, "continued")) return KV_REASON_CONTINUED;
    if (!strcmp(reason, "evict")) return KV_REASON_EVICT;
    if (!strcmp(reason, "shutdown")) return KV_REASON_SHUTDOWN;
    return KV_REASON_UNKNOWN;
}

static kv_cache_options kv_cache_default_options(void) {
    return (kv_cache_options){
        .min_tokens = KV_CACHE_DEFAULT_MIN_TOKENS,
        .cold_max_tokens = KV_CACHE_DEFAULT_COLD_MAX_TOKENS,
        .continued_interval_tokens = KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS,
        .boundary_trim_tokens = KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS,
        .boundary_align_tokens = KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS,
    };
}

static void le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static uint32_t le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t le_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

typedef struct {
    uint32_t h[5];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} sha1_ctx;

static uint32_t rol32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void sha1_transform(sha1_ctx *c, const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++) w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = c->h[0], b = c->h[1], d = c->h[3], e = c->h[4];
    uint32_t cc = c->h[2];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ cc ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & cc) | (b & d) | (cc & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ cc ^ d;
            k = 0xca62c1d6u;
        }
        uint32_t tmp = rol32(a, 5) + f + e + k + w[i];
        e = d;
        d = cc;
        cc = rol32(b, 30);
        b = a;
        a = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += cc;
    c->h[3] += d;
    c->h[4] += e;
}

static void sha1_init(sha1_ctx *c) {
    c->h[0] = 0x67452301u;
    c->h[1] = 0xefcdab89u;
    c->h[2] = 0x98badcfeu;
    c->h[3] = 0x10325476u;
    c->h[4] = 0xc3d2e1f0u;
    c->bytes = 0;
    c->used = 0;
}

static void sha1_update(sha1_ctx *c, const void *ptr, size_t len) {
    const uint8_t *p = ptr;
    c->bytes += len;
    while (len != 0) {
        size_t n = 64 - c->used;
        if (n > len) n = len;
        memcpy(c->block + c->used, p, n);
        c->used += n;
        p += n;
        len -= n;
        if (c->used == 64) {
            sha1_transform(c, c->block);
            c->used = 0;
        }
    }
}

static void sha1_final(sha1_ctx *c, uint8_t out[20]) {
    uint64_t bits = c->bytes * 8;
    uint8_t one = 0x80;
    uint8_t zero = 0;
    sha1_update(c, &one, 1);
    while (c->used != 56) sha1_update(c, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[7 - i] = (uint8_t)(bits >> (8 * i));
    sha1_update(c, len, sizeof(len));
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c->h[i];
    }
}

static void hex20(const uint8_t in[20], char out[41]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 20; i++) {
        out[i * 2] = hex[in[i] >> 4];
        out[i * 2 + 1] = hex[in[i] & 15];
    }
    out[40] = '\0';
}

static void sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    sha1_ctx c;
    sha1_init(&c);
    sha1_update(&c, ptr, len);
    uint8_t digest[20];
    sha1_final(&c, digest);
    hex20(digest, out);
}

static bool id_list_contains(const stop_list *ids, const char *id) {
    if (!ids || !id || !id[0]) return false;
    for (int i = 0; i < ids->len; i++) {
        if (ids->v[i] && !strcmp(ids->v[i], id)) return true;
    }
    return false;
}

static void id_list_push_unique(stop_list *ids, const char *id) {
    if (!ids || !id || !id[0] || id_list_contains(ids, id)) return;
    stop_list_push(ids, xstrdup(id));
}

static void id_list_free(stop_list *ids) {
    stop_list_clear(ids);
    free(ids->v);
    memset(ids, 0, sizeof(*ids));
}

static void collect_tool_call_ids(const chat_msgs *msgs, stop_list *ids) {
    if (!msgs || !ids) return;
    for (int i = 0; i < msgs->len; i++) {
        id_list_push_unique(ids, msgs->v[i].tool_call_id);
        for (int j = 0; j < msgs->v[i].tool_call_ids_len; j++) {
            id_list_push_unique(ids, msgs->v[i].tool_call_ids[j]);
        }
        const tool_calls *calls = &msgs->v[i].calls;
        for (int j = 0; j < calls->len; j++) {
            id_list_push_unique(ids, calls->v[j].id);
        }
    }
}

static bool sha_hex_name(const char *name, char sha[41]) {
    if (strlen(name) != 43 || strcmp(name + 40, ".kv")) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)name[i])) return false;
        sha[i] = (char)tolower((unsigned char)name[i]);
    }
    sha[40] = '\0';
    return true;
}

static char *path_join(const char *dir, const char *name) {
    buf b = {0};
    buf_puts(&b, dir);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') buf_putc(&b, '/');
    buf_puts(&b, name);
    return buf_take(&b);
}

static char *kv_path_for_sha(kv_disk_cache *kc, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return path_join(kc->dir, name);
}

static bool mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

static void kv_entry_free(kv_entry *e) {
    free(e->path);
    memset(e, 0, sizeof(*e));
}

static void kv_cache_clear(kv_disk_cache *kc) {
    for (int i = 0; i < kc->len; i++) kv_entry_free(&kc->entry[i]);
    free(kc->entry);
    kc->entry = NULL;
    kc->len = 0;
    kc->cap = 0;
}

static void kv_cache_push(kv_disk_cache *kc, kv_entry e) {
    if (kc->len == kc->cap) {
        kc->cap = kc->cap ? kc->cap * 2 : 16;
        kc->entry = xrealloc(kc->entry, (size_t)kc->cap * sizeof(kc->entry[0]));
    }
    kc->entry[kc->len++] = e;
}

static const char *find_next_dsml_tool_block(const char *p, const char **end_out) {
    struct block_form {
        const char *start;
        const char *end;
    } forms[] = {
        {"\n\n" DS4_TOOL_CALLS_START, DS4_TOOL_CALLS_END},
        {DS4_TOOL_CALLS_START, DS4_TOOL_CALLS_END},
        {"\n\n" DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT},
        {DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT},
        {"\n\n<tool_calls>", "</tool_calls>"},
        {"<tool_calls>", "</tool_calls>"},
    };

    const char *best = NULL;
    const char *best_end = NULL;
    for (size_t i = 0; i < sizeof(forms) / sizeof(forms[0]); i++) {
        const char *s = strstr(p, forms[i].start);
        if (!s || (best && s >= best)) continue;
        const char *e = strstr(s, forms[i].end);
        if (!e) continue;
        best = s;
        best_end = e + strlen(forms[i].end);
    }
    if (end_out) *end_out = best_end;
    return best;
}

static int tool_memory_count_dsml_in_text(server *s, const char *text) {
    if (!s || s->disable_exact_dsml_tool_replay || !text || !text[0]) return 0;
    int count = 0;
    pthread_mutex_lock(&s->tool_mu);
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_dsml_tool_block(p, &end);
        if (!start || !end) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b) count += b->refs;
        p = end;
    }
    pthread_mutex_unlock(&s->tool_mu);
    return count;
}

static bool kv_tool_map_measure_locked(server *s, const char *text,
                                       uint32_t *count_out,
                                       uint64_t *bytes_out) {
    uint32_t count = 0;
    uint64_t bytes = KV_TOOL_MAP_HEADER;
    uint64_t scan = ++s->tool_mem.scan_clock;
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_dsml_tool_block(p, &end);
        if (!start || !end) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t dsml_len = b->len;
                if (id_len > UINT32_MAX || dsml_len > UINT32_MAX) continue;
                if (count == UINT32_MAX) return false;
                if (UINT64_MAX - bytes < 8u ||
                    UINT64_MAX - bytes - 8u < (uint64_t)id_len ||
                    UINT64_MAX - bytes - 8u - (uint64_t)id_len < (uint64_t)dsml_len)
                    return false;
                count++;
                bytes += 8u + (uint64_t)id_len + (uint64_t)dsml_len;
            }
        }
        p = end;
    }
    if (count == 0) bytes = 0;
    if (count_out) *count_out = count;
    if (bytes_out) *bytes_out = bytes;
    return true;
}

static bool kv_tool_map_serialized_size(server *s, const char *text,
                                        uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!s || s->disable_exact_dsml_tool_replay || !text || !text[0]) return true;

    pthread_mutex_lock(&s->tool_mu);
    bool ok = kv_tool_map_measure_locked(s, text, NULL, bytes_out);
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}

static bool kv_tool_map_write(server *s, FILE *fp, const char *text,
                              uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!s || s->disable_exact_dsml_tool_replay || !fp || !text || !text[0]) return true;

    pthread_mutex_lock(&s->tool_mu);
    uint32_t count = 0;
    uint64_t bytes = 0;
    bool ok = kv_tool_map_measure_locked(s, text, &count, &bytes);
    if (!ok) {
        pthread_mutex_unlock(&s->tool_mu);
        return false;
    }
    if (count == 0) {
        pthread_mutex_unlock(&s->tool_mu);
        return true;
    }

    uint8_t h[KV_TOOL_MAP_HEADER];
    h[0] = KV_TOOL_MAP_MAGIC0;
    h[1] = KV_TOOL_MAP_MAGIC1;
    h[2] = KV_TOOL_MAP_MAGIC2;
    h[3] = KV_TOOL_MAP_VERSION;
    le_put32(h + 4, count);
    ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h);

    uint64_t scan = ++s->tool_mem.scan_clock;
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_dsml_tool_block(p, &end);
        if (!start || !end || !ok) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; ok && e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t dsml_len = b->len;
                if (id_len > UINT32_MAX || dsml_len > UINT32_MAX) continue;
                uint8_t lens[8];
                le_put32(lens, (uint32_t)id_len);
                le_put32(lens + 4, (uint32_t)dsml_len);
                ok = fwrite(lens, 1, sizeof(lens), fp) == sizeof(lens) &&
                     fwrite(e->id, 1, id_len, fp) == id_len &&
                     fwrite(b->dsml, 1, dsml_len, fp) == dsml_len;
            }
        }
        p = end;
    }
    pthread_mutex_unlock(&s->tool_mu);

    if (ok && written_bytes) *written_bytes = bytes;
    return ok;
}

static int kv_tool_map_load_from_pos(server *s, FILE *fp, const stop_list *wanted) {
    if (!s || s->disable_exact_dsml_tool_replay || !fp) return 0;
    uint8_t h[KV_TOOL_MAP_HEADER];
    size_t n = fread(h, 1, sizeof(h), fp);
    if (n == 0 && feof(fp)) return 0;
    if (n != sizeof(h)) return 0;
    if (h[0] != KV_TOOL_MAP_MAGIC0 || h[1] != KV_TOOL_MAP_MAGIC1 ||
        h[2] != KV_TOOL_MAP_MAGIC2 || h[3] != KV_TOOL_MAP_VERSION) return 0;

    uint32_t count = le_get32(h + 4);
    if ((uint64_t)count > (uint64_t)tool_memory_max_entries(&s->tool_mem) * 4u) return 0;
    int loaded = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint8_t lens[8];
        if (fread(lens, 1, sizeof(lens), fp) != sizeof(lens)) return loaded;
        uint32_t id_len = le_get32(lens);
        uint32_t dsml_len = le_get32(lens + 4);
        if (id_len == 0 || id_len > 256 || dsml_len == 0 ||
            dsml_len > DS4_TOOL_MEMORY_MAX_BYTES) return loaded;
        char *id = xmalloc((size_t)id_len + 1);
        char *dsml = xmalloc((size_t)dsml_len + 1);
        bool ok = fread(id, 1, id_len, fp) == id_len &&
                  fread(dsml, 1, dsml_len, fp) == dsml_len;
        id[id_len] = '\0';
        dsml[dsml_len] = '\0';
        if (ok && (!wanted || id_list_contains(wanted, id))) {
            tool_memory_put_source(s, id, dsml, TOOL_MEMORY_DISK);
            loaded++;
        }
        free(id);
        free(dsml);
        if (!ok) return loaded;
    }
    return loaded;
}

static void kv_fill_header(uint8_t h[KV_CACHE_FIXED_HEADER], uint8_t quant_bits,
                           uint8_t reason, uint8_t ext_flags,
                           uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                           uint64_t created_at, uint64_t last_used,
                           uint64_t payload_bytes) {
    memset(h, 0, KV_CACHE_FIXED_HEADER);
    h[0] = KV_CACHE_MAGIC0;
    h[1] = KV_CACHE_MAGIC1;
    h[2] = KV_CACHE_MAGIC2;
    h[3] = KV_CACHE_VERSION;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    le_put32(h + 8, tokens);
    le_put32(h + 12, hits);
    le_put32(h + 16, ctx_size);
    le_put64(h + 24, created_at);
    le_put64(h + 32, last_used);
    le_put64(h + 40, payload_bytes);
}

static bool kv_read_header(FILE *fp, kv_entry *e, uint32_t *text_bytes) {
    uint8_t h[KV_CACHE_FIXED_HEADER];
    if (fread(h, 1, sizeof(h), fp) != sizeof(h)) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 ||
        h[2] != KV_CACHE_MAGIC2 || h[3] != KV_CACHE_VERSION) return false;
    e->quant_bits = h[4];
    e->reason = h[5] <= KV_REASON_SHUTDOWN ? h[5] : KV_REASON_UNKNOWN;
    e->ext_flags = h[6];
    e->tokens = le_get32(h + 8);
    e->hits = le_get32(h + 12);
    e->ctx_size = le_get32(h + 16);
    e->created_at = le_get64(h + 24);
    e->last_used = le_get64(h + 32);
    e->payload_bytes = le_get64(h + 40);
    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) return false;
    *text_bytes = le_get32(tb);
    e->text_bytes = *text_bytes;
    return e->tokens != 0 && (e->quant_bits == 2 || e->quant_bits == 4);
}

static bool kv_read_entry_file(const char *path, const char sha[41], kv_entry *out) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < (off_t)(KV_CACHE_FIXED_HEADER + 4)) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    kv_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &e, &text_bytes);
    fclose(fp);
    if (!ok) return false;
    const uint64_t fixed = KV_CACHE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < (uint64_t)text_bytes ||
        UINT64_MAX - fixed - (uint64_t)text_bytes < e.payload_bytes) return false;
    const uint64_t expected = fixed + (uint64_t)text_bytes + e.payload_bytes;
    if ((uint64_t)st.st_size < expected) return false;
    memcpy(e.sha, sha, 41);
    e.path = xstrdup(path);
    e.file_size = (uint64_t)st.st_size;
    *out = e;
    return true;
}

static void kv_cache_refresh(kv_disk_cache *kc) {
    if (!kc->enabled) return;
    kv_cache_clear(kc);
    DIR *d = opendir(kc->dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!sha_hex_name(de->d_name, sha)) continue;
        char *path = path_join(kc->dir, de->d_name);
        kv_entry e = {0};
        if (kv_read_entry_file(path, sha, &e)) kv_cache_push(kc, e);
        free(path);
    }
    closedir(d);
}

static bool kv_cache_touch_file(const char *path, uint32_t hits) {
    FILE *fp = fopen(path, "r+b");
    if (!fp) return false;
    kv_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &e, &text_bytes);
    if (ok) {
        uint8_t h[KV_CACHE_FIXED_HEADER];
        uint64_t now = (uint64_t)time(NULL);
        kv_fill_header(h, e.quant_bits, e.reason, e.ext_flags, e.tokens, hits, e.ctx_size,
                       e.created_at, now, e.payload_bytes);
        ok = fseek(fp, 0, SEEK_SET) == 0 &&
             fwrite(h, 1, sizeof(h), fp) == sizeof(h);
    }
    fclose(fp);
    return ok;
}

static void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs) {
    if (!s || s->disable_exact_dsml_tool_replay || !s->kv.enabled || !msgs) return;
    stop_list wanted = {0};
    collect_tool_call_ids(msgs, &wanted);
    if (wanted.len == 0) return;

    DIR *d = opendir(s->kv.dir);
    if (!d) {
        id_list_free(&wanted);
        return;
    }
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!sha_hex_name(de->d_name, sha)) continue;
        (void)sha;
        char *path = path_join(s->kv.dir, de->d_name);
        FILE *fp = fopen(path, "rb");
        free(path);
        if (!fp) continue;

        kv_entry hdr = {0};
        uint32_t text_bytes = 0;
        bool ok = kv_read_header(fp, &hdr, &text_bytes);
        uint64_t skip = (uint64_t)text_bytes + hdr.payload_bytes;
        if (ok && (hdr.ext_flags & KV_EXT_TOOL_MAP) &&
            skip <= (uint64_t)INT64_MAX &&
            fseeko(fp, (off_t)skip, SEEK_CUR) == 0)
        {
            kv_tool_map_load_from_pos(s, fp, &wanted);
        }
        fclose(fp);
    }
    closedir(d);
    id_list_free(&wanted);
}

static double kv_entry_eviction_score(const kv_entry *e, const ds4_tokens *live,
                                      const char *protected_sha,
                                      uint64_t now) {
    if (!e || e->file_size == 0) return 0.0;
    (void)live;
    if (protected_sha && !strcmp(e->sha, protected_sha)) {
        /* The store path calls eviction immediately after renaming the new
         * checkpoint into place.  Without a protected score, a full cache can
         * choose that brand-new zero-hit file as the cheapest single deletion,
         * turning the save into pure I/O waste.  Treat it as maximally valuable
         * for this eviction pass.  If it is the only remaining entry and the
         * budget is still exceeded, the loop will still delete it below. */
        return DBL_MAX;
    }
    /*
     * Hits count successful disk reuses, but a fresh snapshot is still useful:
     * it may be the only copy of the session that is about to be evicted from
     * RAM.  Continued checkpoints are deliberately aligned restart frontiers,
     * so do not demote them just because they are a prefix of the live session.
     * Use hits+1 for eviction value so a just-written checkpoint does not get
     * deleted immediately just because its persisted hit counter is still 0.
     *
     * Fold time-since-last_used into the hit bonus so a workload change does
     * not let once-popular files outrank live prefixes forever.  Effective
     * hits halve every KV_CACHE_HIT_HALF_LIFE_SECONDS of inactivity.  A fresh
     * hit rewrites last_used, so matching prompts re-boost the entry.  Once
     * the bonus is below one hundredth of a hit, treat it as zero; this lets
     * the last_used tie-breaker evict stale files against equally dense fresh
     * checkpoints instead of preserving a microscopic floating-point bonus.
     */
    double effective_hits = (double)e->hits;
    uint64_t used_at = e->last_used ? e->last_used : e->created_at;
    if (used_at == 0) {
        effective_hits = 0.0;
    } else if (now > used_at) {
        double elapsed = (double)(now - used_at);
        effective_hits *= exp2(-elapsed / (double)KV_CACHE_HIT_HALF_LIFE_SECONDS);
        if (effective_hits < KV_CACHE_MIN_EFFECTIVE_HITS) effective_hits = 0.0;
    }
    return (effective_hits + 1.0) * (double)e->tokens / (double)e->file_size;
}

static void kv_cache_evict(kv_disk_cache *kc, const ds4_tokens *live,
                           const char *protected_sha) {
    if (!kc->enabled || kc->budget_bytes == 0) return;
    kv_cache_refresh(kc);
    const uint64_t now = (uint64_t)time(NULL);
    uint64_t total = 0;
    for (int i = 0; i < kc->len; i++) total += kc->entry[i].file_size;
    if (protected_sha) {
        uint64_t protected_size = 0;
        for (int i = 0; i < kc->len; i++) {
            if (!strcmp(kc->entry[i].sha, protected_sha)) {
                protected_size = kc->entry[i].file_size;
                break;
            }
        }
        /* Do not preserve an entry that cannot fit in the budget by itself:
         * protecting it would first delete every older checkpoint, then delete
         * the new one anyway. */
        if (protected_size > kc->budget_bytes) protected_sha = NULL;
    }
    while (total > kc->budget_bytes && kc->len > 0) {
        int victim = 0;
        double victim_score = kv_entry_eviction_score(&kc->entry[0], live, protected_sha, now);
        for (int i = 1; i < kc->len; i++) {
            double score = kv_entry_eviction_score(&kc->entry[i], live, protected_sha, now);
            if (score < victim_score ||
                (score == victim_score && kc->entry[i].last_used < kc->entry[victim].last_used))
            {
                victim = i;
                victim_score = score;
            }
        }
        kv_entry e = kc->entry[victim];
        if (unlink(e.path) == 0) {
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache evicted reason=disk-cache-full tokens=%u hits=%u size=%.2f MiB file=%s",
                       e.tokens,
                       e.hits,
                       (double)e.file_size / (1024.0 * 1024.0),
                       e.path ? e.path : "?");
            if (total >= e.file_size) total -= e.file_size;
            else total = 0;
        } else {
            total = 0;
        }
        kv_entry_free(&e);
        memmove(kc->entry + victim, kc->entry + victim + 1,
                (size_t)(kc->len - victim - 1) * sizeof(kc->entry[0]));
        kc->len--;
    }
}

static bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb,
                          bool reject_different_quant, kv_cache_options opt) {
    memset(kc, 0, sizeof(*kc));
    if (!dir) return false;
    if (!mkdir_p(dir)) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to create KV cache directory %s: %s", dir, strerror(errno));
        return false;
    }
    kc->enabled = true;
    kc->dir = xstrdup(dir);
    if (budget_mb == 0) budget_mb = KV_CACHE_DEFAULT_MB;
    kc->budget_bytes = budget_mb * 1024ull * 1024ull;
    kc->reject_different_quant = reject_different_quant;
    kc->opt = opt;
    kv_cache_evict(kc, NULL, NULL);
    server_log(DS4_LOG_KVCACHE,
               "ds4-server: KV disk cache %s (budget=%llu MiB, cross-quant=%s, min=%d, cold_max=%d, continued=%d, trim=%d, align=%d, hit_half_life=%llus)",
               kc->dir,
               (unsigned long long)(kc->budget_bytes / (1024ull * 1024ull)),
               reject_different_quant ? "reject" : "accept",
               kc->opt.min_tokens,
               kc->opt.cold_max_tokens,
               kc->opt.continued_interval_tokens,
               kc->opt.boundary_trim_tokens,
               kc->opt.boundary_align_tokens,
               (unsigned long long)KV_CACHE_HIT_HALF_LIFE_SECONDS);
    return true;
}

static void kv_cache_close(kv_disk_cache *kc) {
    kv_cache_clear(kc);
    free(kc->dir);
    memset(kc, 0, sizeof(*kc));
}

static char *render_tokens_text(ds4_engine *engine, const ds4_tokens *tokens, size_t *out_len) {
    buf b = {0};
    for (int i = 0; i < tokens->len; i++) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, tokens->v[i], &len);
        buf_append(&b, piece, len);
        free(piece);
    }
    if (out_len) *out_len = b.len;
    return buf_take(&b);
}

static bool byte_prefix_match(const char *text, size_t text_len,
                              const char *prefix, size_t prefix_len) {
    return prefix_len <= text_len &&
           (prefix_len == 0 || memcmp(text, prefix, prefix_len) == 0);
}

static const char *kv_cache_key_kind(uint8_t ext_flags) {
    if (ext_flags & KV_EXT_RESPONSES_VISIBLE) return "responses-visible";
    if (ext_flags & KV_EXT_THINKING_VISIBLE) return "thinking-visible";
    return "token-text";
}

static void tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n) {
    dst->len = 0;
    if (n > src->len) n = src->len;
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, src->v[i]);
}

static void tokens_append(ds4_tokens *dst, const ds4_tokens *src) {
    if (!dst || !src) return;
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

static void build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out)
{
    ds4_tokens_copy(out, exact_prefix);

    ds4_tokens suffix = {0};
    /* The suffix may start with DS4 chat markers such as <｜User｜> or
     * </think>, so use the rendered-chat tokenizer, not plain text BPE. */
    ds4_tokenize_rendered_chat(engine, suffix_text ? suffix_text : "", &suffix);
    tokens_append(out, &suffix);
    ds4_tokens_free(&suffix);
}

static int kv_cache_store_len(const kv_disk_cache *kc, int tokens) {
    const int trim = kc->opt.boundary_trim_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (tokens > kc->opt.min_tokens + trim) {
        int stable = tokens - trim;
        if (align > 0) stable -= stable % align;
        if (stable >= kc->opt.min_tokens) return stable;
    }
    return tokens;
}

static int kv_cache_chat_anchor_pos(const kv_disk_cache *kc,
                                    const ds4_tokens *prompt,
                                    int user_token_id,
                                    int assistant_token_id) {
    if (!prompt || user_token_id < 0 || assistant_token_id < 0) return -1;

    /* Cold checkpoints are meant to maximize reuse across independent agent
     * sessions.  The stable part of a rendered chat is everything before the
     * user message that asks the model to do this specific task.  Some clients
     * put stable user-role scaffolding first, for example Codex renders:
     *
     *   <｜User｜><environment_context>...</environment_context><｜User｜>task
     *
     * So use the last user marker before the first assistant marker, not the
     * first one.  Stop at the first assistant marker so multi-turn histories do
     * not turn old conversation content into low-value cold cache files; live
     * and continued checkpoints cover same-session continuation instead.
     *
     * The returned value is a token prefix length and deliberately excludes the
     * selected <｜User｜> token.  This is an exact special-token boundary, so the
     * generic trim/align heuristic is not needed to avoid tokenizer merges. */
    int last_user = -1;
    for (int i = 0; i < prompt->len; i++) {
        const int token = prompt->v[i];
        if (token == assistant_token_id) break;
        if (token == user_token_id) last_user = i;
    }
    return last_user >= kc->opt.min_tokens ? last_user : -1;
}

static int kv_cache_continued_step(const kv_disk_cache *kc) {
    if (!kc->enabled || kc->opt.continued_interval_tokens <= 0) return 0;
    int step = kc->opt.continued_interval_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (align > 0) {
        step = ((step + align - 1) / align) * align;
        if (step <= 0) step = align;
    }
    return step;
}

static int kv_cache_continued_store_target(const kv_disk_cache *kc, int live_tokens) {
    const int step = kv_cache_continued_step(kc);
    if (step <= 0) return 0;
    if (live_tokens < kc->opt.min_tokens) return 0;

    /* Continued checkpoints cannot roll the live graph backward.  Save only
     * at absolute aligned frontiers, not relative to the last cold/evict file.
     * Otherwise an early cold checkpoint can shift the whole schedule and leave
     * long generations with no recent durable restart point. */
    if (live_tokens % step != 0) return 0;
    if (live_tokens <= kc->continued_last_store_tokens) return 0;
    return live_tokens;
}

/* A same-text-prefix file can be reused by a larger context, but not by a
 * smaller one: the payload was validated against the context capacity recorded
 * in the file.  If the existing file cannot be used by this server, replace it
 * so this context can still populate its own cache. */
static bool kv_cache_file_text_matches(const char *path, const char sha[41],
                                       const char *text, size_t text_len) {
    if (text_len > UINT32_MAX) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;

    kv_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &hdr, &text_bytes) &&
              text_bytes == (uint32_t)text_len;
    char *stored = NULL;
    if (ok) {
        stored = xmalloc((size_t)text_bytes + 1);
        ok = fread(stored, 1, text_bytes, fp) == text_bytes;
    }
    fclose(fp);
    if (!ok) {
        free(stored);
        return false;
    }

    char stored_sha[41];
    sha1_bytes_hex(stored, text_bytes, stored_sha);
    ok = !strcmp(stored_sha, sha) &&
         (text_len == 0 || memcmp(stored, text, text_len) == 0);
    free(stored);
    return ok;
}

static bool kv_cache_file_size_bytes(uint64_t text_bytes,
                                     uint64_t payload_bytes,
                                     uint64_t tool_map_bytes,
                                     uint64_t *file_bytes) {
    const uint64_t fixed = KV_CACHE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < text_bytes ||
        UINT64_MAX - fixed - text_bytes < payload_bytes ||
        UINT64_MAX - fixed - text_bytes - payload_bytes < tool_map_bytes)
        return false;
    if (file_bytes) *file_bytes = fixed + text_bytes + payload_bytes + tool_map_bytes;
    return true;
}

static bool kv_cache_budget_required(uint64_t file_bytes,
                                     uint64_t *required_bytes) {
    /* The serialized size is deterministic for one snapshot, including the
     * optional tool map.  We still reserve 1% headroom so filesystem/accounting
     * surprises or a concurrently-added tool mapping cannot produce a file that
     * is immediately removed by the cache budget pass. */
    uint64_t slack = file_bytes / 100u;
    if (file_bytes % 100u) slack++;
    if (UINT64_MAX - file_bytes < slack) return false;
    if (required_bytes) *required_bytes = file_bytes + slack;
    return true;
}

static bool kv_cache_file_size_fits(const kv_disk_cache *kc,
                                    uint64_t text_bytes,
                                    uint64_t payload_bytes,
                                    uint64_t tool_map_bytes,
                                    uint64_t *file_bytes_out,
                                    uint64_t *required_bytes_out) {
    uint64_t file_bytes = 0;
    if (!kv_cache_file_size_bytes(text_bytes, payload_bytes, tool_map_bytes,
                                  &file_bytes))
        return false;
    if (file_bytes_out) *file_bytes_out = file_bytes;
    if (!kc || kc->budget_bytes == 0) return true;
    uint64_t required = 0;
    if (!kv_cache_budget_required(file_bytes, &required)) return false;
    if (required_bytes_out) *required_bytes_out = required;
    return required <= kc->budget_bytes;
}

static bool kv_cache_existing_compatible(kv_disk_cache *kc, const char *path,
                                         const char sha[41],
                                         const char *text, size_t text_len,
                                         int quant_bits, int ctx_size) {
    if (access(path, F_OK) != 0) return false;
    kv_entry e = {0};
    if (!kv_read_entry_file(path, sha, &e)) return false;
    bool compatible = (!kc->reject_different_quant || e.quant_bits == (uint8_t)quant_bits) &&
                      e.ctx_size <= (uint32_t)ctx_size &&
                      kv_cache_file_text_matches(path, sha, text, text_len);
    kv_entry_free(&e);
    if (!compatible) {
        if (unlink(path) == 0) {
            server_log(DS4_LOG_KVCACHE, "ds4-server: kv cache replaced incompatible file %s", path);
        }
        return false;
    }
    return true;
}

static void kv_cache_rewrite_tool_map(server *s, const char *path, const char *text) {
    if (!s || !path || !text || tool_memory_count_dsml_in_text(s, text) == 0) return;
    FILE *fp = fopen(path, "r+b");
    if (!fp) return;
    kv_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &hdr, &text_bytes);
    uint64_t end = KV_CACHE_FIXED_HEADER + 4ull + (uint64_t)text_bytes + hdr.payload_bytes;
    if (ok && end <= (uint64_t)INT64_MAX &&
        fseeko(fp, (off_t)end, SEEK_SET) == 0 &&
        ftruncate(fileno(fp), (off_t)end) == 0)
    {
        uint64_t ignored = 0;
        ok = kv_tool_map_write(s, fp, text, &ignored) && fflush(fp) == 0;
        if (ok && ignored > 0) {
            uint8_t h[KV_CACHE_FIXED_HEADER];
            uint64_t now = (uint64_t)time(NULL);
            kv_fill_header(h, hdr.quant_bits, hdr.reason,
                           (uint8_t)(hdr.ext_flags | KV_EXT_TOOL_MAP),
                           hdr.tokens, hdr.hits, hdr.ctx_size,
                           hdr.created_at, now, hdr.payload_bytes);
            ok = fseeko(fp, 0, SEEK_SET) == 0 &&
                 fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
                 fflush(fp) == 0;
        }
    }
    fclose(fp);
    (void)ok;
}

static bool kv_cache_store_live_prefix_text(server *s, const ds4_tokens *tokens,
                                            int store_len, const char *reason,
                                            const char *cache_text_override,
                                            uint8_t cache_text_ext,
                                            const char *cache_text_key) {
    kv_disk_cache *kc = &s->kv;
    if (!kc->enabled) return false;
    if (!tokens || store_len < kc->opt.min_tokens) return false;
    const int original_len = tokens->len;

    ds4_tokens store_tokens = {0};
    tokens_copy_prefix(&store_tokens, tokens, store_len);

    const int quant_bits = ds4_engine_routed_quant_bits(s->engine);
    if (quant_bits != 2 && quant_bits != 4) {
        ds4_tokens_free(&store_tokens);
        return false;
    }
    char err[160] = {0};
    /* Disk cache persistence must observe the graph exactly as-is.  If callers
     * want a shorter prefix, they first prefill to that prefix and only then call
     * this function.  This keeps cache population from doing hidden inference. */
    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens ||
        live_tokens->len != store_tokens.len ||
        !ds4_tokens_starts_with(live_tokens, &store_tokens))
    {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d",
                   store_tokens.len,
                   reason,
                   live_tokens ? live_tokens->len : -1);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    uint64_t payload_bytes = ds4_session_payload_bytes(s->session);
    if (payload_bytes == 0) {
        ds4_tokens_free(&store_tokens);
        return false;
    }

    size_t text_len = 0;
    char *text = NULL;
    const bool text_override = cache_text_override && cache_text_override[0];
    if (text_override) {
        text = xstrdup(cache_text_override);
        text_len = strlen(text);
    } else {
        text = render_tokens_text(s->engine, &store_tokens, &text_len);
    }
    if (text_len > UINT32_MAX) {
        server_log(DS4_LOG_KVCACHE, "ds4-server: kv cache skipped tokens=%d because rendered text is too large", store_tokens.len);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    uint64_t tool_map_est_bytes = 0;
    if (!kv_tool_map_serialized_size(s, text, &tool_map_est_bytes)) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache skipped tokens=%d reason=%s because tool map size overflowed",
                   store_tokens.len, reason);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    uint64_t est_file_bytes = 0, est_required_bytes = 0;
    if (!kv_cache_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                 tool_map_est_bytes,
                                 &est_file_bytes, &est_required_bytes)) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache skipped tokens=%d reason=%s because estimated file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB",
                   store_tokens.len,
                   reason,
                   (double)est_file_bytes / (1024.0 * 1024.0),
                   (double)est_required_bytes / (1024.0 * 1024.0),
                   (double)kc->budget_bytes / (1024.0 * 1024.0));
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    char sha[41];
    sha1_bytes_hex(text, text_len, sha);
    char *path = kv_path_for_sha(kc, sha);

    if (kv_cache_existing_compatible(kc, path, sha, text, text_len,
                                     quant_bits, ds4_session_ctx(s->session))) {
        kv_cache_rewrite_tool_map(s, path, text);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return true;
    }

    buf tmpb = {0};
    buf_printf(&tmpb, "%s.tmp.%ld", path, (long)getpid());
    char *tmp = buf_take(&tmpb);
    const double save_t0 = now_sec();
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        server_log(DS4_LOG_KVCACHE, "ds4-server: kv cache failed to create %s: %s save=%.1f ms",
                   tmp, strerror(errno), (now_sec() - save_t0) * 1000.0);
        free(tmp);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    const uint64_t now = (uint64_t)time(NULL);
    uint8_t h[KV_CACHE_FIXED_HEADER];
    uint8_t ext_flags = tool_map_est_bytes > 0 ? KV_EXT_TOOL_MAP : 0;
    if (text_override) ext_flags |= cache_text_ext;
    kv_fill_header(h, (uint8_t)quant_bits, kv_reason_code(reason), ext_flags,
                   (uint32_t)store_tokens.len, 0,
                   (uint32_t)ds4_session_ctx(s->session), now, now, payload_bytes);
    uint8_t tb[4];
    le_put32(tb, (uint32_t)text_len);
    uint64_t tool_map_bytes = 0;
    errno = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len &&
              ds4_session_save_payload(s->session, fp, err, sizeof(err)) == 0 &&
              kv_tool_map_write(s, fp, text, &tool_map_bytes) &&
              fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    uint64_t final_file_bytes = 0, final_required_bytes = 0;
    bool final_size_over_budget = false;
    if (ok && !kv_cache_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                       tool_map_bytes,
                                       &final_file_bytes, &final_required_bytes))
    {
        final_size_over_budget = true;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    const double save_ms = (now_sec() - save_t0) * 1000.0;
    if (!ok) {
        if (final_size_over_budget) {
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache skipped tokens=%d reason=%s because final file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB save=%.1f ms",
                       store_tokens.len,
                       reason,
                       (double)final_file_bytes / (1024.0 * 1024.0),
                       (double)final_required_bytes / (1024.0 * 1024.0),
                       (double)kc->budget_bytes / (1024.0 * 1024.0),
                       save_ms);
        } else {
            server_log(DS4_LOG_KVCACHE, "ds4-server: kv cache store failed (%s): %s save=%.1f ms",
                       reason,
                       saved_errno ? strerror(saved_errno) : (err[0] ? err : "unknown error"),
                       save_ms);
        }
        unlink(tmp);
    } else {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache stored tokens=%d trimmed=%d reason=%s key=%s size=%.2f MiB save=%.1f ms",
                   store_tokens.len,
                   original_len - store_tokens.len,
                   reason,
                   text_override ? (cache_text_key ? cache_text_key : "visible-transcript") : "token-text",
                   (double)(KV_CACHE_FIXED_HEADER + 4ull + text_len + payload_bytes + tool_map_bytes) / (1024.0 * 1024.0),
                   save_ms);
        kv_cache_evict(kc, live_tokens, sha);
    }
    free(tmp);
    free(text);
    free(path);
    ds4_tokens_free(&store_tokens);
    return ok;
}

static bool kv_cache_store_live_prefix(server *s, const ds4_tokens *tokens,
                                       int store_len, const char *reason) {
    return kv_cache_store_live_prefix_text(s, tokens, store_len, reason,
                                           NULL, 0, NULL);
}

static void kv_cache_store_current(server *s, const char *reason) {
    const ds4_tokens *tokens = ds4_session_tokens(s->session);
    if (!tokens) return;

    char *visible_text = NULL;
    uint8_t visible_ext = 0;
    const char *visible_key = NULL;
    pthread_mutex_lock(&s->tool_mu);
    if (s->responses_live.valid &&
        s->responses_live.live_tokens == tokens->len &&
        s->responses_live.visible_text &&
        s->responses_live.visible_text[0])
    {
        visible_text = xstrdup(s->responses_live.visible_text);
        visible_ext = KV_EXT_RESPONSES_VISIBLE;
        visible_key = "responses-visible";
    } else if (s->thinking_live.valid &&
               s->thinking_live.live_tokens == tokens->len &&
               s->thinking_live.visible_text &&
               s->thinking_live.visible_text[0])
    {
        visible_text = xstrdup(s->thinking_live.visible_text);
        visible_ext = KV_EXT_THINKING_VISIBLE;
        visible_key = "thinking-visible";
    }
    pthread_mutex_unlock(&s->tool_mu);

    /* A visible live checkpoint can contain hidden reasoning that the client
     * intentionally does not replay.  For disk recovery after a session switch,
     * key that payload by the visible protocol transcript, not by rendering the
     * hidden sampled tokens.  On load, DS4 restores the hidden KV payload and
     * tokenizes only the visible suffix that follows this key. */
    if (visible_text) {
        kv_cache_store_live_prefix_text(s, tokens, tokens->len, reason,
                                        visible_text, visible_ext, visible_key);
        free(visible_text);
    } else {
        kv_cache_store_live_prefix(s, tokens, tokens->len, reason);
    }
}

static void kv_cache_note_store(kv_disk_cache *kc, int tokens) {
    if (tokens > kc->continued_last_store_tokens) {
        kc->continued_last_store_tokens = tokens;
    }
}

static int kv_cache_suppress_continued_store(kv_disk_cache *kc, int tokens) {
    if (kv_cache_continued_store_target(kc, tokens) != tokens) return -1;
    int old = kc->continued_last_store_tokens;
    kv_cache_note_store(kc, tokens);
    return old;
}

static void kv_cache_restore_suppressed_continued(kv_disk_cache *kc,
                                                  int old_tokens,
                                                  int suppressed_tokens) {
    if (old_tokens >= 0 && kc->continued_last_store_tokens == suppressed_tokens) {
        kc->continued_last_store_tokens = old_tokens;
    }
}

static void kv_cache_maybe_store_continued(server *s) {
    kv_disk_cache *kc = &s->kv;
    const ds4_tokens *tokens = ds4_session_tokens(s->session);
    if (!tokens) return;
    const int target = kv_cache_continued_store_target(kc, tokens->len);
    if (target == 0) return;
    if (kv_cache_store_live_prefix(s, tokens, target, "continued")) {
        kv_cache_note_store(kc, target);
    }
}

static int kv_cache_find_text_prefix(kv_disk_cache *kc, const char *prompt_text,
                                     int quant_bits, int ctx_size) {
    if (!prompt_text) return -1;
    const size_t prompt_bytes = strlen(prompt_text);
    kv_cache_refresh(kc);
    int best = -1;
    for (int i = 0; i < kc->len; i++) {
        kv_entry *e = &kc->entry[i];
        if (e->text_bytes > prompt_bytes || e->text_bytes > SIZE_MAX) continue;
        if ((int)e->tokens < kc->opt.min_tokens) continue;
        if ((uint32_t)ctx_size < e->ctx_size) continue;
        if (kc->reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
        if (best >= 0) {
            kv_entry *b = &kc->entry[best];
            if (e->text_bytes < b->text_bytes) continue;
            if (e->text_bytes == b->text_bytes && e->tokens <= b->tokens) continue;
        }
        char sha[41];
        sha1_bytes_hex(prompt_text, (size_t)e->text_bytes, sha);
        if (!strcmp(sha, e->sha)) best = i;
    }
    return best;
}

static int kv_cache_try_load_text(server *s, const char *prompt_text,
                                  ds4_tokens *effective_prompt,
                                  char **loaded_path_out,
                                  uint8_t *loaded_ext_flags_out,
                                  bool responses_protocol) {
    if (loaded_path_out) *loaded_path_out = NULL;
    if (loaded_ext_flags_out) *loaded_ext_flags_out = 0;
    if (effective_prompt) effective_prompt->len = 0;
    kv_disk_cache *kc = &s->kv;
    if (!kc->enabled || !prompt_text) return 0;
    const int quant_bits = ds4_engine_routed_quant_bits(s->engine);
    if (quant_bits != 2 && quant_bits != 4) return 0;
    const size_t prompt_bytes = strlen(prompt_text);
    int idx = kv_cache_find_text_prefix(kc, prompt_text, quant_bits,
                                        ds4_session_ctx(s->session));
    if (idx < 0) return 0;

    kv_entry e = kc->entry[idx];
    char *path = xstrdup(e.path);
    const double load_t0 = now_sec();
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return 0;
    }
    uint32_t text_bytes = 0;
    kv_entry hdr = {0};
    const char *fail_reason = "invalid header";
    bool header_ok = kv_read_header(fp, &hdr, &text_bytes);
    char *cached_text = NULL;
    if (header_ok) {
        if ((uint64_t)text_bytes > prompt_bytes) {
            header_ok = false;
            fail_reason = "cached text is longer than prompt";
        } else {
            cached_text = xmalloc((size_t)text_bytes + 1);
            if (fread(cached_text, 1, text_bytes, fp) != text_bytes) {
                header_ok = false;
                fail_reason = "truncated cached text";
            } else {
                cached_text[text_bytes] = '\0';
                char text_sha[41];
                sha1_bytes_hex(cached_text, text_bytes, text_sha);
                if (strcmp(text_sha, e.sha)) {
                    header_ok = false;
                    fail_reason = "cached text hash mismatch";
                } else if (!byte_prefix_match(prompt_text, prompt_bytes,
                                              cached_text, text_bytes)) {
                    header_ok = false;
                    fail_reason = "cached text prefix mismatch";
                }
            }
        }
    }
    char err[160] = {0};
    int loaded = 0;
    if (header_ok && ds4_session_load_payload(s->session, fp, hdr.payload_bytes, err, sizeof(err)) == 0) {
        const ds4_tokens *loaded_tokens = ds4_session_tokens(s->session);
        if (loaded_tokens && loaded_tokens->len == (int)hdr.tokens) {
            loaded = (int)hdr.tokens;
            if (effective_prompt) {
                /* The cache lookup was by bytes, but the graph state is still
                 * the exact token history stored in the payload.  Build the
                 * prompt we give ds4_session_sync() from that exact history and
                 * tokenize only the text suffix after the byte prefix. */
                build_prompt_from_exact_prefix_and_text_suffix(
                    s->engine, loaded_tokens, prompt_text + text_bytes,
                    effective_prompt);
            }
            if (hdr.ext_flags & KV_EXT_TOOL_MAP) kv_tool_map_load_from_pos(s, fp, NULL);
        } else {
            ds4_session_invalidate(s->session);
            unlink(path);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache discarded corrupt text-prefix payload%s%s %s",
                       responses_protocol ? " " : "",
                       responses_protocol ? "RESPPROTO" : "",
                       path);
        }
    } else {
        if (header_ok) ds4_session_invalidate(s->session);
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache load failed%s%s %s: %s load=%.1f ms",
                   responses_protocol ? " " : "",
                   responses_protocol ? "RESPPROTO" : "",
                   path,
                   header_ok ? err : fail_reason,
                   (now_sec() - load_t0) * 1000.0);
    }
    fclose(fp);

    if (loaded > 0) {
        const double load_ms = (now_sec() - load_t0) * 1000.0;
        if (loaded_path_out) *loaded_path_out = xstrdup(path);
        if (loaded_ext_flags_out) *loaded_ext_flags_out = hdr.ext_flags;
        kc->continued_last_store_tokens = loaded;
        const char *key_kind = kv_cache_key_kind(hdr.ext_flags);
        if (kc->opt.cold_max_tokens > 0 && loaded > kc->opt.cold_max_tokens) {
            unlink(path);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms consumed file=%s",
                       responses_protocol ? " " : "",
                       responses_protocol ? "RESPPROTO" : "",
                       loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        } else {
            kv_cache_touch_file(path, hdr.hits + 1);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms file=%s",
                       responses_protocol ? " " : "",
                       responses_protocol ? "RESPPROTO" : "",
                       loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        }
    }
    free(cached_text);
    free(path);
    return loaded;
}

static int kv_cache_try_load(server *s, const request *req,
                             ds4_tokens *effective_prompt,
                             char **loaded_path_out,
                             uint8_t *loaded_ext_flags_out) {
    return kv_cache_try_load_text(s, req ? req->prompt_text : NULL,
                                  effective_prompt,
                                  loaded_path_out,
                                  loaded_ext_flags_out,
                                  req && req->api == API_RESPONSES);
}

static int live_text_prefix_prompt(server *s, const request *req,
                                   ds4_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len <= 0) return 0;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, live_tokens, &live_text_len);
    const size_t prompt_text_len = strlen(req->prompt_text);
    if (!byte_prefix_match(req->prompt_text, prompt_text_len,
                           live_text, live_text_len))
    {
        free(live_text);
        return 0;
    }

    /* This is the core text-prefix case.  The live graph is authoritative, so
     * keep its sampled tokenization and tokenize only the request bytes that
     * come after it.  Reusing req->prompt's token suffix would be wrong: full
     * prompt BPE may have merged across this byte boundary. */
    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + live_text_len,
        effective_prompt);
    free(live_text);
    return live_tokens->len;
}

/* Tool-output-only Responses continuation.
 *
 * Some clients send just the new tool outputs after a tool call.  There is no
 * long visible prefix to match in that shape; the call_id itself is the
 * protocol binding to the previous live assistant output.  Use it only when the
 * remembered live frontier and call-id set match exactly. */
static int responses_live_continuation_prompt(server *s, const request *req,
                                              int live_pos,
                                              ds4_tokens *effective_prompt,
                                              int *matched_ids) {
    if (!s || !req || !effective_prompt) return 0;
    if (req->api != API_RESPONSES || !req->responses_live_suffix_text) return 0;
    if (req->responses_live_call_ids.len == 0) return 0;
    if (!responses_live_matches_request(s, &req->responses_live_call_ids,
                                        live_pos)) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->responses_live_suffix_text,
        effective_prompt);
    if (matched_ids) *matched_ids = req->responses_live_call_ids.len;
    return live_tokens->len;
}

/* Tool-result Anthropic continuation.
 *
 * /v1/messages has no server-side response object like the OpenAI Responses
 * API, but its tool_use_id is still a precise continuation handle inside a live
 * local agent loop.  When the IDs and live token frontier match, continue from
 * the sampled DSML state and append only the user tool_result suffix. */
static int anthropic_live_continuation_prompt(server *s, const request *req,
                                              int live_pos,
                                              ds4_tokens *effective_prompt,
                                              int *matched_ids) {
    if (!s || !req || !effective_prompt) return 0;
    if (req->api != API_ANTHROPIC || !req->anthropic_live_suffix_text) return 0;
    if (req->anthropic_live_call_ids.len == 0) return 0;
    if (!anthropic_live_matches_request(s, &req->anthropic_live_call_ids,
                                        live_pos)) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->anthropic_live_suffix_text,
        effective_prompt);
    if (matched_ids) *matched_ids = req->anthropic_live_call_ids.len;
    return live_tokens->len;
}

/* Visible-replay Responses continuation.
 *
 * Other clients send the full visible transcript on every turn even though the
 * API semantics still make the request a continuation.  For Responses, exact
 * token-prefix matching is the wrong first question: hidden reasoning may be
 * live in KV but absent from the replay by design.  Instead, verify that the
 * request's rendered text begins with the visible transcript remembered at the
 * live frontier.  If it does, continue from the live token prefix and tokenize
 * only the bytes after that visible boundary.
 *
 * If this check fails, DS4 has no special Responses state to trust.  The caller
 * then uses normal token/text/disk matching, which is the correct fallback for
 * cold starts, edits, restarts, or cross-client replays. */
static int responses_live_visible_prefix_prompt(server *s, const request *req,
                                                int live_pos,
                                                ds4_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    if (req->api != API_RESPONSES) return 0;

    const size_t prompt_len = strlen(req->prompt_text);
    size_t visible_len = 0;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = s->responses_live.valid &&
              s->responses_live.live_tokens == live_pos &&
              s->responses_live.visible_text &&
              s->responses_live.visible_len < prompt_len &&
              byte_prefix_match(req->prompt_text, prompt_len,
                                s->responses_live.visible_text,
                                s->responses_live.visible_len);
    if (ok) visible_len = s->responses_live.visible_len;
    pthread_mutex_unlock(&s->tool_mu);
    if (!ok) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + visible_len,
        effective_prompt);
    return live_tokens->len;
}

/* Tool-less thinking continuation.
 *
 * Chat/completions and Anthropic do not have a previous_response_id object that
 * binds a later request to the last sampled turn.  Still, after a normal
 * tool-less thinking answer, the next prompt renderer intentionally omits that
 * hidden reasoning.  The live KV state is richer than the visible transcript.
 *
 * Remembering the visible transcript as a key lets us keep the sampled hidden
 * KV when the next request clearly extends that same visible history.  This is
 * the same byte-prefix idea used by the disk cache: the client-visible text
 * selects the checkpoint, while the payload stays the exact sampled token
 * frontier.  If the visible key does not match, callers fall back to ordinary
 * token/text/disk matching. */
static int thinking_live_visible_prefix_prompt(server *s, const request *req,
                                               int live_pos,
                                               ds4_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    if (req->kind != REQ_CHAT || req->api == API_RESPONSES) return 0;

    const size_t prompt_len = strlen(req->prompt_text);
    size_t visible_len = 0;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = s->thinking_live.valid &&
              s->thinking_live.live_tokens == live_pos &&
              s->thinking_live.visible_text &&
              s->thinking_live.visible_len < prompt_len &&
              byte_prefix_match(req->prompt_text, prompt_len,
                                s->thinking_live.visible_text,
                                s->thinking_live.visible_len);
    if (ok) visible_len = s->thinking_live.visible_len;
    pthread_mutex_unlock(&s->tool_mu);
    if (!ok) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + visible_len,
        effective_prompt);
    return live_tokens->len;
}

