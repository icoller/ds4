static char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls) {
    buf suffix = {0};
    if (ds4_think_mode_enabled(r->think_mode)) {
        buf_puts(&suffix, reasoning ? reasoning : "");
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, content ? content : "");
    append_dsml_tool_calls_text(&suffix, calls);
    buf_puts(&suffix, "<｜end▁of▁sentence｜>");
    return buf_take(&suffix);
}

static char *build_responses_visible_assistant_suffix(const request *r,
                                                      const char *content,
                                                      const char *reasoning,
                                                      const tool_calls *calls) {
    buf suffix = {0};
    /* This suffix mirrors what a Responses client can replay, not necessarily
     * every token in KV.  Hidden reasoning stays live in the session unless the
     * next client replay is expected to include it.  In practice, pi replays
     * reasoning summaries for tool-call turns, but not for final assistant
     * answers; Codex currently requests no summaries at all.  So only include
     * reasoning in the remembered visible prefix when this assistant turn ended
     * in tool calls.  A client that does replay final-answer reasoning will not
     * match this visible shortcut and can still use exact token-prefix replay. */
    if (ds4_think_mode_enabled(r->think_mode)) {
        if (r->reasoning_summary_emit && calls && calls->len > 0) {
            buf_puts(&suffix, reasoning ? reasoning : "");
        }
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, content ? content : "");
    append_dsml_tool_calls_text(&suffix, calls);
    buf_puts(&suffix, "<｜end▁of▁sentence｜>");
    return buf_take(&suffix);
}

/* In thinking mode without tools, old assistant reasoning is intentionally not
 * rendered back into later prompts.  The sampled live graph still contains the
 * reasoning bytes, so the next request would miss the session cache even though
 * the visible conversation prefix is logically the same.
 *
 *   prompt-without-final-<think> + </think> + visible-content + eos
 *
 * is exactly the visible prefix that render_chat_prompt_text() will produce on
 * the next turn.  Do not rebuild the KV cache to erase hidden reasoning here:
 * that caused long post-answer pauses and threw away useful sampled state.
 * Instead, remember the visible bytes as a key for the current sampled frontier.
 * The next request can then continue from live KV while tokenizing only the new
 * visible suffix. */
static char *build_toolless_thinking_visible_text(const request *r,
                                                  const char *content) {
    if (!r || !r->prompt_text) return NULL;
    if (!ds4_think_mode_enabled(r->think_mode)) return NULL;

    size_t pt_len = strlen(r->prompt_text);
    const char *think_tag = "<think>";
    size_t tag_len = strlen(think_tag);
    if (pt_len < tag_len ||
        memcmp(r->prompt_text + pt_len - tag_len, think_tag, tag_len) != 0) {
        return NULL;
    }

    buf visible = {0};
    buf_append(&visible, r->prompt_text, pt_len - tag_len);
    buf_puts(&visible, "</think>");
    buf_puts(&visible, content ? content : "");
    buf_puts(&visible, "<｜end▁of▁sentence｜>");
    return buf_take(&visible);
}

static void remember_thinking_checkpoint(server *s, const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content) {
    char *visible = build_toolless_thinking_visible_text(&j->req, content);
    if (!visible) return;

    thinking_live_remember(s, visible);
    server_log(DS4_LOG_KVCACHE,
               "ds4-server: thinking live checkpoint remembered ctx=%s live=%d visible=%zu",
               ctx, ds4_session_pos(s->session), strlen(visible));
    trace_event(s, trace_id,
                "thinking live checkpoint remembered: live=%d visible=%zu",
                ds4_session_pos(s->session), strlen(visible));
    free(visible);
}

/* After a successful tool-call finish, make the live checkpoint match what the
 * next request will render.  Usually that is just the exact DSML remembered by
 * tool id.  If a client sends a tool call without an id we know, the fallback
 * renderer still builds valid DSML from JSON, and this function either rewrites
 * the short suffix in place or reloads an older disk checkpoint before replay. */
static void canonicalize_tool_checkpoint(server *s, const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content,
                                         const char *reasoning, const tool_calls *calls) {
    if (!calls || calls->len == 0 || !j->req.prompt_text) return;

    char *suffix_text = build_tool_checkpoint_suffix(&j->req, content, reasoning, calls);

    buf rendered = {0};
    buf_puts(&rendered, j->req.prompt_text);
    buf_puts(&rendered, suffix_text);

    ds4_tokens canonical = {0};
    ds4_tokenize_rendered_chat(s->engine, rendered.ptr ? rendered.ptr : "", &canonical);
    const int live_len = ds4_session_pos(s->session);
    const int common = ds4_session_common_prefix(s->session, &canonical);
    if (common == live_len && canonical.len == live_len) goto done;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, ds4_session_tokens(s->session), &live_text_len);
    if (live_text_len == rendered.len &&
        (live_text_len == 0 || memcmp(live_text, rendered.ptr, live_text_len) == 0))
    {
        /* The graph already represents the bytes the next request will render.
         * Token-level canonicalization would only replace a valid sampled
         * history with a different BPE spelling of the same transcript. */
        free(live_text);
        goto done;
    }
    free(live_text);

    if (common < j->req.prompt.len) {
        trace_event(s, trace_id,
                    "tool checkpoint canonicalization skipped: common=%d prompt=%d live=%d canonical=%d",
                    common, j->req.prompt.len, live_len, canonical.len);
        goto done;
    }

    char err[160] = {0};
    ds4_session_rewrite_result rr =
        ds4_session_rewrite_from_common(s->session, &canonical, common,
                                        err, sizeof(err));
    if (rr == DS4_SESSION_REWRITE_OK) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d",
                   ctx, common, live_len, canonical.len);
        trace_event(s, trace_id,
                    "tool checkpoint canonicalized: common=%d live=%d canonical=%d",
                    common, live_len, canonical.len);
    } else if (rr == DS4_SESSION_REWRITE_REBUILD_NEEDED) {
        /* The generated DSML suffix and the canonical prompt share a prefix,
         * but the generated tail is too large to overwrite safely inside the
         * live raw-window ring.  Prefer an older disk checkpoint over replaying
         * a very long conversation from token zero. */
        char *path = NULL;
        ds4_tokens effective = {0};
        int loaded = kv_cache_try_load_text(s, rendered.ptr ? rendered.ptr : "",
                                            &effective, &path, NULL, false);
        if (loaded == 0) ds4_session_invalidate(s->session);

        char sync_err[160] = {0};
        const ds4_tokens *sync_prompt = loaded > 0 ? &effective : &canonical;
        char rebuild_ctx[48];
        request_ctx_span(rebuild_ctx, sizeof(rebuild_ctx), loaded, sync_prompt->len);
        int replay_tokens = sync_prompt->len - loaded;
        if (replay_tokens < 0) replay_tokens = sync_prompt->len;
        int canonical_tail_tokens = canonical.len - common;
        if (canonical_tail_tokens < 0) canonical_tail_tokens = canonical.len;
        int discarded_live_tokens = live_len - common;
        if (discarded_live_tokens < 0) discarded_live_tokens = 0;
        const char *source = loaded > 0 ? "disk" : "full";
        const double rebuild_t0 = now_sec();
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalization needs %d tokens rebuild ctx=%s request_ctx=%s reason=canonical-tail-rewrite tail=%d discard=%d common=%d live=%d target=%d cached=%d source=%s%s%s",
                   replay_tokens,
                   rebuild_ctx,
                   ctx,
                   canonical_tail_tokens,
                   discarded_live_tokens,
                   common,
                   live_len,
                   canonical.len,
                   loaded,
                   source,
                   path ? " file=" : "",
                   path ? path : "");
        server_prefill_progress rebuild_progress = {
            .srv = s,
            .fd = j->fd,
            .req = &j->req,
            .kind = j->req.kind,
            .prompt_tokens = sync_prompt->len,
            .cached_tokens = loaded,
            .phase = "tool checkpoint rebuild",
            .has_tools = j->req.has_tools,
            .t0 = rebuild_t0,
        };
        snprintf(rebuild_progress.ctx, sizeof(rebuild_progress.ctx), "%s", rebuild_ctx);
        ds4_session_set_progress(s->session, server_progress_cb, &rebuild_progress);
        if (ds4_session_sync(s->session, sync_prompt, sync_err, sizeof(sync_err)) == 0) {
            ds4_session_set_progress(s->session, NULL, NULL);
            const double rebuild_sec = now_sec() - rebuild_t0;
            if (loaded > 0) {
                server_log(DS4_LOG_KVCACHE,
                           "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=disk cached=%d replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, loaded, replay_tokens, canonical.len, rebuild_sec);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via disk: common=%d live=%d canonical=%d cached=%d file=%s",
                            common, live_len, canonical.len, loaded, path ? path : "");
            } else {
                server_log(DS4_LOG_KVCACHE,
                           "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=full cached=0 replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, replay_tokens, canonical.len, rebuild_sec);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via rebuild: common=%d live=%d canonical=%d reason=%s",
                            common, live_len, canonical.len, err);
            }
        } else {
            ds4_session_set_progress(s->session, NULL, NULL);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: tool checkpoint rebuild failed ctx=%s request_ctx=%s source=%s cached=%d replay=%d target=%d error=\"%s\"",
                       rebuild_ctx, ctx, source, loaded, replay_tokens,
                       canonical.len, sync_err);
            trace_event(s, trace_id, "tool checkpoint canonicalization failed after rebuild request: %s", sync_err);
        }
        ds4_tokens_free(&effective);
        free(path);
    } else {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"",
                   ctx, common, live_len, canonical.len, err);
        trace_event(s, trace_id, "tool checkpoint canonicalization failed: %s", err);
    }

done:
    ds4_tokens_free(&canonical);
    buf_free(&rendered);
    free(suffix_text);
}

static bool should_canonicalize_tool_checkpoint(const server *s, const tool_calls *calls) {
    if (!calls || calls->len == 0) return false;
    if (s && !s->disable_exact_dsml_tool_replay &&
        calls->raw_dsml && calls->raw_dsml[0])
    {
        return false;
    }
    return true;
}

#define DS4_CONTINUATION_MAX_ENTRIES 128

static void continuation_link_tail(server *s, continuation_entry *e) {
    e->prev = s->continuation_tail;
    e->next = NULL;
    if (s->continuation_tail) s->continuation_tail->next = e;
    else s->continuation_head = e;
    s->continuation_tail = e;
}

static void continuation_unlink(server *s, continuation_entry *e) {
    if (e->prev) e->prev->next = e->next;
    else s->continuation_head = e->next;
    if (e->next) e->next->prev = e->prev;
    else s->continuation_tail = e->prev;
    e->prev = e->next = NULL;
}

static void continuation_touch(server *s, continuation_entry *e) {
    if (!s || !e || s->continuation_tail == e) return;
    continuation_unlink(s, e);
    continuation_link_tail(s, e);
}

static void continuation_entry_free(continuation_entry *e) {
    if (!e) return;
    free(e->response_id);
    free(e->visible_text);
    free(e->model);
    free(e->implicit_session_key);
    ds4_session_snapshot_free(&e->snapshot);
    free(e);
}

static continuation_entry *continuation_find_by_id(server *s, const char *id) {
    if (!s || !s->continuations_by_id || !id || !id[0]) return NULL;
    void *v = raxFind(s->continuations_by_id, (unsigned char *)id, strlen(id));
    return v == raxNotFound ? NULL : v;
}

static continuation_entry *continuation_find_by_implicit_key(server *s, const char *key) {
    if (!s || !s->continuation_alias_by_key || !key || !key[0]) return NULL;
    void *v = raxFind(s->continuation_alias_by_key, (unsigned char *)key, strlen(key));
    return v == raxNotFound ? NULL : v;
}

static void continuation_remove(server *s, continuation_entry *e) {
    if (!s || !e) return;
    if (s->continuations_by_id && e->response_id) {
        void *old = NULL;
        (void)raxRemove(s->continuations_by_id, (unsigned char *)e->response_id,
                        strlen(e->response_id), &old);
    }
    if (s->continuation_alias_by_key && e->implicit_session_key) {
        void *cur = raxFind(s->continuation_alias_by_key,
                            (unsigned char *)e->implicit_session_key,
                            strlen(e->implicit_session_key));
        if (cur != raxNotFound && cur == e) {
            void *old = NULL;
            (void)raxRemove(s->continuation_alias_by_key,
                            (unsigned char *)e->implicit_session_key,
                            strlen(e->implicit_session_key), &old);
        }
    }
    continuation_unlink(s, e);
    if (s->continuation_entries > 0) s->continuation_entries--;
    continuation_entry_free(e);
}

static void continuation_clear_all(server *s) {
    while (s && s->continuation_head) continuation_remove(s, s->continuation_head);
}

static void request_finalize_implicit_session_key(request *r) {
    if (!r || r->api != API_RESPONSES) return;
    free(r->implicit_session_key);
    r->implicit_session_key = NULL;
    const bool has_session = r->client_session_key && r->client_session_key[0];
    const bool has_basis = r->implicit_session_basis && r->implicit_session_basis[0];
    if (!has_session && !has_basis) return;

    buf raw = {0};
    buf_puts(&raw, "api:responses\nmodel:");
    buf_puts(&raw, r->model ? r->model : "");
    buf_puts(&raw, "\nsession:");
    buf_puts(&raw, r->client_session_key ? r->client_session_key : "");
    buf_puts(&raw, "\nbasis:");
    buf_puts(&raw, r->implicit_session_basis ? r->implicit_session_basis : "");
    if (raw.len == 0) {
        buf_free(&raw);
        return;
    }
    char sha[41];
    sha1_bytes_hex(raw.ptr ? raw.ptr : "", raw.len, sha);
    buf_free(&raw);

    buf key = {0};
    buf_puts(&key, "resp-implicit:");
    buf_puts(&key, sha);
    r->implicit_session_key = buf_take(&key);
}

static char *responses_build_visible_transcript(const request *r,
                                                const char *content,
                                                const char *reasoning,
                                                const tool_calls *calls) {
    if (!r || !r->prompt_text) return NULL;
    char *suffix = build_responses_visible_assistant_suffix(r, content, reasoning, calls);
    buf out = {0};
    buf_puts(&out, r->prompt_text);
    buf_puts(&out, suffix ? suffix : "");
    free(suffix);
    return buf_take(&out);
}

static int continuation_restore_prompt(server *s, const continuation_entry *entry,
                                       const char *suffix_text,
                                       ds4_tokens *effective_prompt,
                                       char *err, size_t errlen) {
    if (!s || !entry || !suffix_text || !effective_prompt) return 0;
    int loaded = 0;
    char *loaded_path = NULL;
    if (entry->visible_text && s->kv.enabled) {
        loaded = kv_cache_try_load_text(s, entry->visible_text, NULL,
                                        &loaded_path, NULL, true);
        free(loaded_path);
    }
    if (loaded == 0 && entry->snapshot.ptr && entry->snapshot.len > 0) {
        if (ds4_session_load_snapshot(s->session, &entry->snapshot, err, errlen) != 0) {
            return 0;
        }
        const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
        loaded = live_tokens ? live_tokens->len : 0;
    }
    if (loaded <= 0) {
        snprintf(err, errlen, "stored continuation state is not available");
        return 0;
    }

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != loaded) {
        snprintf(err, errlen, "stored continuation state is inconsistent");
        return 0;
    }
    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, suffix_text, effective_prompt);
    return loaded;
}

static void continuation_store_response(server *s, const request *r,
                                        const char *response_id,
                                        const char *visible_text) {
    if (!s || !r || r->api != API_RESPONSES || !response_id || !response_id[0]) return;

    if (!s->continuations_by_id) s->continuations_by_id = raxNew();
    if (!s->continuation_alias_by_key) s->continuation_alias_by_key = raxNew();
    if (!s->continuations_by_id || !s->continuation_alias_by_key) die("out of memory");

    continuation_entry *e = continuation_find_by_id(s, response_id);
    if (!e) {
        e = xmalloc(sizeof(*e));
        memset(e, 0, sizeof(*e));
        e->response_id = xstrdup(response_id);
        continuation_link_tail(s, e);
        s->continuation_entries++;
        if (!raxInsert(s->continuations_by_id, (unsigned char *)e->response_id,
                       strlen(e->response_id), e, NULL))
        {
            die("out of memory");
        }
    } else {
        if (s->continuation_alias_by_key && e->implicit_session_key) {
            void *cur = raxFind(s->continuation_alias_by_key,
                                (unsigned char *)e->implicit_session_key,
                                strlen(e->implicit_session_key));
            if (cur != raxNotFound && cur == e) {
                void *old = NULL;
                (void)raxRemove(s->continuation_alias_by_key,
                                (unsigned char *)e->implicit_session_key,
                                strlen(e->implicit_session_key), &old);
            }
        }
        free(e->visible_text);
        e->visible_text = NULL;
        free(e->model);
        e->model = NULL;
        free(e->implicit_session_key);
        e->implicit_session_key = NULL;
        ds4_session_snapshot_free(&e->snapshot);
        continuation_touch(s, e);
    }

    e->created_at = e->created_at ? e->created_at : (uint64_t)time(NULL);
    e->last_used = (uint64_t)time(NULL);
    e->model = r->model ? xstrdup(r->model) : NULL;
    e->visible_text = visible_text ? xstrdup(visible_text) : NULL;
    if (r->implicit_session_key && r->implicit_session_key[0]) {
        e->implicit_session_key = xstrdup(r->implicit_session_key);
        void *old = NULL;
        (void)raxRemove(s->continuation_alias_by_key,
                        (unsigned char *)e->implicit_session_key,
                        strlen(e->implicit_session_key), &old);
        if (!raxInsert(s->continuation_alias_by_key,
                       (unsigned char *)e->implicit_session_key,
                       strlen(e->implicit_session_key), e, NULL))
        {
            die("out of memory");
        }
    }

    bool disk_backed = false;
    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (e->visible_text && live_tokens && s->kv.enabled &&
        live_tokens->len >= s->kv.opt.min_tokens)
    {
        disk_backed = kv_cache_store_live_prefix_text(
            s, live_tokens, live_tokens->len, "continued",
            e->visible_text, KV_EXT_RESPONSES_VISIBLE, "responses-visible");
    }
    if (!disk_backed) {
        char snap_err[160] = {0};
        if (ds4_session_save_snapshot(s->session, &e->snapshot,
                                      snap_err, sizeof(snap_err)) != 0) {
            server_log(DS4_LOG_WARNING,
                       "ds4-server: failed to save continuation snapshot for %s: %s",
                       response_id, snap_err);
        }
    }

    while (s->continuation_entries > DS4_CONTINUATION_MAX_ENTRIES &&
           s->continuation_head)
    {
        continuation_remove(s, s->continuation_head);
    }
}

/* Execute one request on the worker-owned session.
 *
 * Clients resend full prompts as text.  The worker first tries the old exact
 * token-prefix hit, then a rendered-text prefix hit for the live checkpoint,
 * then disk text-prefix restart snapshots, then a cold prefill.  On text-prefix
 * hits we build a fresh effective prompt from the checkpoint's exact token
 * history plus a newly tokenized string suffix; the canonical full-prompt
 * tokens are not sliced because BPE may merge across the byte boundary.  Cold
 * prompt caching is handled before generation: if the stable checkpoint is
 * shorter than the full prompt, we prefill to that boundary, store it, and
 * immediately continue to the real prompt.  The live graph therefore always
 * moves forward. */
