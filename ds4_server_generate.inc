static void generate_job(server *s, job *j) {
    char err[160];
    err[0] = '\0';
    const int old_pos = ds4_session_pos(s->session);
    const int common = ds4_session_common_prefix(s->session, &j->req.prompt);
    trace_cache_diag cache_diag = {0};
    trace_cache_capture(&cache_diag, ds4_session_tokens(s->session),
                        &j->req.prompt, old_pos, common);
    ds4_tokens effective_prompt = {0};
    const ds4_tokens *prompt_for_sync = &j->req.prompt;
    const bool responses_protocol = j->req.api == API_RESPONSES;
    bool responses_live_continuation = false;
    bool responses_stored_continuation = false;
    bool anthropic_live_continuation = false;
    bool thinking_live_continuation = false;
    const char *responses_live_match = NULL;
    int responses_live_match_ids = 0;
    int anthropic_live_match_ids = 0;
    int cached = 0;
    const char *cache_source = "none";
    continuation_entry *stored_continuation = NULL;
    const char *stored_continuation_match = NULL;
    if (responses_protocol && j->req.continuation_suffix_text) {
        if (j->req.previous_response_id && j->req.previous_response_id[0]) {
            stored_continuation = continuation_find_by_id(s, j->req.previous_response_id);
            stored_continuation_match = "previous_response_id";
        }
        if (!stored_continuation &&
            j->req.implicit_session_key && j->req.implicit_session_key[0]) {
            stored_continuation = continuation_find_by_implicit_key(
                s, j->req.implicit_session_key);
            stored_continuation_match = "implicit_session";
        }
        if (stored_continuation) {
            cached = continuation_restore_prompt(s, stored_continuation,
                                                 j->req.continuation_suffix_text,
                                                 &effective_prompt,
                                                 err, sizeof(err));
            if (cached > 0) {
                responses_stored_continuation = true;
                prompt_for_sync = &effective_prompt;
                cache_source = "responses-snapshot";
            }
        }
    }
    /* Responses gets the first chance to continue from live state.  This is
     * the whole point of the API shape: a request that is bound to prior live
     * output by visible transcript or tool call ids does not need to prove an
     * exact token-prefix match.  Exact token/text/disk matching remains the
     * fallback when the live state is absent or no longer describes the
     * request. */
    if (!responses_stored_continuation) {
        cached = responses_live_visible_prefix_prompt(s, &j->req, old_pos,
                                                      &effective_prompt);
    }
    cache_source = cached > 0 ? "responses-visible" : "none";
    if (responses_stored_continuation) {
        cache_source = "responses-snapshot";
    } else if (cached > 0) {
        responses_live_match = "visible-prefix";
        if (responses_live_matches_request(s, &j->req.responses_live_call_ids,
                                           old_pos))
        {
            responses_live_match_ids = j->req.responses_live_call_ids.len;
        }
    }
    if (!responses_stored_continuation && cached == 0) {
        cached = responses_live_continuation_prompt(s, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &responses_live_match_ids);
        cache_source = cached > 0 ? "responses-tool-output" : "none";
        if (cached > 0) responses_live_match = "tool-output-ids";
    }
    if (responses_stored_continuation || cached > 0) {
        responses_live_continuation = true;
        prompt_for_sync = &effective_prompt;
    } else {
        cached = anthropic_live_continuation_prompt(s, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &anthropic_live_match_ids);
        if (cached > 0) {
            anthropic_live_continuation = true;
            cache_source = "anthropic-tool-output";
            prompt_for_sync = &effective_prompt;
        }
    }
    if (!responses_stored_continuation &&
        cached == 0 && responses_protocol &&
        j->req.responses_requires_live_tool_state)
    {
        /* The parser saw a valid live call_id, but by worker execution time the
         * live frontier no longer matches.  Since the request did not replay
         * the prior assistant call, there is no stateless prefix to match and
         * no disk key to search by. */
        ds4_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "Responses continuation state is not available; retry by replaying the full input history");
        return;
    } else if (!responses_stored_continuation &&
               cached == 0 && responses_protocol &&
               j->req.previous_response_id && j->req.previous_response_id[0])
    {
        ds4_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "previous_response_id state is not available; replay full input or retry with the latest response");
        return;
    } else if (cached == 0 && j->req.api == API_ANTHROPIC &&
               j->req.anthropic_requires_live_tool_state)
    {
        ds4_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "Anthropic continuation state is not available; retry by replaying the full messages history");
        return;
    } else if (cached == 0) {
        cached = common == old_pos && j->req.prompt.len >= old_pos ? common : 0;
        cache_source = cached > 0 ? "memory-token" : "none";
    }
    if (cached == 0) {
        int thinking_cached =
            thinking_live_visible_prefix_prompt(s, &j->req, old_pos,
                                                &effective_prompt);
        if (thinking_cached > 0) {
            cached = thinking_cached;
            cache_source = "thinking-visible";
            thinking_live_continuation = true;
            prompt_for_sync = &effective_prompt;
        }
    }
    int disk_cached = 0;
    char *disk_cache_path = NULL;
    uint8_t disk_cache_ext_flags = 0;
    if (cached == 0) {
        int text_cached = live_text_prefix_prompt(s, &j->req, &effective_prompt);
        if (text_cached > 0) {
            cached = text_cached;
            cache_source = "memory-text";
            prompt_for_sync = &effective_prompt;
        }
    }
    if (cached == 0 && old_pos > 0) {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: live kv cache miss%s live=%d prompt=%d common=%d reason=%s",
                   responses_protocol ? " RESPPROTO" : "",
                   old_pos, j->req.prompt.len, common,
                   trace_cache_miss_reason(&cache_diag));
    }
    if (cached == 0) s->kv.continued_last_store_tokens = 0;
    if (s->kv.enabled && cached == 0 && old_pos >= s->kv.opt.min_tokens) {
        /* Loading a disk snapshot replaces the live Metal session.  Persist the
         * current checkpoint first, otherwise a cache hit for an older prefix
         * would silently discard the newer conversation state. */
        kv_cache_store_current(s, "evict");
    }
    if (cached == 0) {
        disk_cached = kv_cache_try_load(s, &j->req, &effective_prompt,
                                        &disk_cache_path,
                                        &disk_cache_ext_flags);
        if (disk_cached > 0) {
            cached = disk_cached;
            cache_source = "disk-text";
            prompt_for_sync = &effective_prompt;
        }
    }
    const bool responses_reasoning_state_preserved =
        cached > 0 &&
        ((!strcmp(cache_source, "responses-visible") ||
          !strcmp(cache_source, "responses-tool-output")) ||
         (!strcmp(cache_source, "disk-text") &&
          (disk_cache_ext_flags & KV_EXT_RESPONSES_VISIBLE)));
    const bool responses_visible_replay_without_reasoning =
        responses_protocol &&
        j->req.responses_requires_live_reasoning &&
        !responses_reasoning_state_preserved;
    const int prompt_tokens = prompt_for_sync->len;
    /* OpenAI usage details: the reusable prefix is a cache read, while the
     * effective prompt suffix evaluated by ds4_session_sync() is written into
     * the live KV cache and can be reused by the next request. */
    j->req.cache_read_tokens = cached;
    j->req.cache_write_tokens = prompt_tokens > cached ? prompt_tokens - cached : 0;

    const double t0 = now_sec();
    uint64_t trace_id = trace_begin(s, j, cached, prompt_tokens, &cache_diag,
                                    cache_source, disk_cached, disk_cache_path);
    free(disk_cache_path);
    char ctx_span[48];
    request_ctx_span(ctx_span, sizeof(ctx_span), cached, prompt_tokens);
    server_prefill_progress progress = {
        .srv = s,
        .fd = j->fd,
        .req = &j->req,
        .kind = j->req.kind,
        .prompt_tokens = prompt_tokens,
        .cached_tokens = cached,
        .has_tools = j->req.has_tools,
        .responses_protocol = responses_protocol,
        .t0 = t0,
    };
    snprintf(progress.ctx, sizeof(progress.ctx), "%s", ctx_span);
    char req_flags[64];
    log_flags(req_flags, sizeof(req_flags), responses_protocol,
              j->req.has_tools, false, false, false);
    if (responses_stored_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: responses stored continuation RESPPROTO match=%s cached=%d prompt=%d",
                   stored_continuation_match ? stored_continuation_match : "snapshot",
                   cached,
                   prompt_tokens);
    } else if (responses_live_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: responses live continuation RESPPROTO match=%s ids=%d cached=%d prompt=%d",
                   responses_live_match ? responses_live_match : "unknown",
                   responses_live_match_ids,
                   cached,
                   prompt_tokens);
    } else if (anthropic_live_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: anthropic live continuation match=tool-output-ids ids=%d cached=%d prompt=%d",
                   anthropic_live_match_ids,
                   cached,
                   prompt_tokens);
    } else if (thinking_live_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: thinking live continuation match=visible-prefix cached=%d prompt=%d",
                   cached,
                   prompt_tokens);
    }
    if (responses_visible_replay_without_reasoning) {
        /* The request replays a prior tool-call turn but omits the hidden
         * reasoning that originally led to it.  A live Responses checkpoint, or
         * a responses-visible disk checkpoint, would preserve that hidden KV.
         * If neither is available, continue from the visible transcript instead
         * of surfacing a hard error to the user.  This is lower fidelity, but it
         * lets old / restarted agent sessions recover and is exactly what the
         * client asked us to prefill. */
        server_log(DS4_LOG_WARNING,
                   "ds4-server: responses replay RESPPROTO missing reasoning state; continuing from visible history source=%s cached=%d prompt=%d",
                   cache_source,
                   cached,
                   prompt_tokens);
        trace_event(s, trace_id,
                    "responses replay missing reasoning state; continuing from visible history source=%s cached=%d",
                    cache_source, cached);
    }
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s prompt start",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               ctx_span,
               req_flags[0] ? " " : "",
               req_flags);
    ds4_session_set_progress(s->session, server_progress_cb, &progress);

    int cold_store_len = 0;
    if (cached == 0 &&
        s->kv.enabled &&
        prompt_for_sync->len >= s->kv.opt.min_tokens &&
        s->kv.opt.cold_max_tokens > 0 &&
        prompt_for_sync->len <= s->kv.opt.cold_max_tokens)
    {
        const int anchor = kv_cache_chat_anchor_pos(&s->kv, prompt_for_sync,
                                                    ds4_token_user(s->engine),
                                                    ds4_token_assistant(s->engine));
        cold_store_len = anchor >= s->kv.opt.min_tokens ?
                         anchor : kv_cache_store_len(&s->kv, prompt_for_sync->len);
    }
    int suppressed_continued_last = -1;
    if (cold_store_len >= s->kv.opt.min_tokens) {
        /* A cold checkpoint can land exactly on the continued-checkpoint
         * frontier.  The prefill progress callback would then write the same
         * prefix as "continued" while we are intentionally stopping there to
         * write it as "cold".  Mark the frontier as already handled before the
         * sync reaches it; if the cold write fails, restore the old schedule so
         * a later continued write can still try. */
        suppressed_continued_last =
            kv_cache_suppress_continued_store(&s->kv, cold_store_len);
    }

    if (s->kv.enabled &&
        cold_store_len >= s->kv.opt.min_tokens &&
        cold_store_len < prompt_for_sync->len)
    {
        ds4_tokens prefix = {0};
        tokens_copy_prefix(&prefix, prompt_for_sync, cold_store_len);
        if (ds4_session_sync(s->session, &prefix, err, sizeof(err)) != 0) {
            ds4_tokens_free(&prefix);
            ds4_tokens_free(&effective_prompt);
            ds4_session_set_progress(s->session, NULL, NULL);
            kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                                  cold_store_len);
            trace_event(s, trace_id, "prefill failed: %s", err);
            http_error(j->fd, s->enable_cors, 500, err);
            return;
        }
        if (kv_cache_store_live_prefix(s, prompt_for_sync, cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, cold_store_len);
            suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                                  cold_store_len);
            suppressed_continued_last = -1;
        }
        ds4_tokens_free(&prefix);
    }

    if (ds4_session_sync(s->session, prompt_for_sync, err, sizeof(err)) != 0) {
        ds4_tokens_free(&effective_prompt);
        ds4_session_set_progress(s->session, NULL, NULL);
        kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                              cold_store_len);
        trace_event(s, trace_id, "prefill failed: %s", err);
        http_error(j->fd, s->enable_cors, 500, err);
        return;
    }
    /* Once a non-live request wins, old protocol live bindings are stale. Keep
     * a binding only when this request explicitly continued from it. */
    if (!responses_live_continuation) responses_live_clear(s);
    if (!anthropic_live_continuation) anthropic_live_clear(s);
    if (!thinking_live_continuation) thinking_live_clear(s);
    ds4_session_set_progress(s->session, NULL, NULL);
    kv_cache_maybe_store_continued(s);
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s prompt done %.3fs",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               ctx_span,
               req_flags[0] ? " " : "",
               req_flags,
               now_sec() - t0);
    if (cold_store_len == prompt_for_sync->len) {
        if (kv_cache_store_live_prefix(s, prompt_for_sync, cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, cold_store_len);
            suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                                  cold_store_len);
        }
    }
    char id[96];
    snprintf(id, sizeof(id), "%s-%llu",
             j->req.kind == REQ_CHAT ? "chatcmpl" : "cmpl",
             (unsigned long long)++s->seq);
    char responses_response_id[40] = {0};
    if (responses_protocol) {
        responses_random_id(responses_response_id, sizeof(responses_response_id), "resp_");
    }

    bool structured_stream = request_uses_structured_stream(&j->req);
    anthropic_stream anthropic_live = {0};
    openai_stream openai_live = {0};
    responses_stream responses_live = {0};
    const bool openai_live_chat = request_uses_openai_live_stream(&j->req);
    const bool responses_live_chat = request_uses_responses_live_stream(&j->req);
    long responses_created_at = (long)time(NULL);
    if (j->req.stream) {
        if (!sse_headers(j->fd, s->enable_cors)) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s sse headers failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       req_flags[0] ? " " : "",
                       req_flags);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        if (j->req.api == API_ANTHROPIC &&
            !anthropic_sse_start_live(j->fd, &j->req, id,
                                      prompt_tokens, &anthropic_live)) {
            server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s anthropic stream start failed", ctx_span);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        if (j->req.api == API_OPENAI && j->req.kind == REQ_CHAT &&
            !sse_chunk(j->fd, &j->req, id, NULL, NULL)) {
            server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s openai role chunk failed", ctx_span);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        if (openai_live_chat) openai_stream_start(&j->req, &openai_live);
        if (responses_live_chat) {
            responses_stream_init(&j->req, &responses_live, responses_response_id);
            responses_live.active = true;
            if (!responses_sse_created(j->fd, &j->req, &responses_live, responses_created_at)) {
                server_log(DS4_LOG_GENERATION,
                           "ds4-server: chat ctx=%s%s%s responses created event failed",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags);
                responses_stream_free(&responses_live);
                ds4_tokens_free(&effective_prompt);
                return;
            }
        }
    }

    buf text = {0};
    size_t plain_stream_pos = 0;
    size_t stop_scan_from = 0;
    const char *finish = "length";
    int completion = 0;
    int max_tokens = j->req.max_tokens;
    int room = ds4_session_ctx(s->session) - ds4_session_pos(s->session);
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    bool saw_orphan_tool_end = false;
    size_t tool_scan_from = 0;
    int next_tool_progress = 128;
    int next_decode_log = 50;
    uint64_t rng = j->req.seed ? j->req.seed :
        (((uint64_t)time(NULL) << 32) ^ ((uint64_t)s->seq << 1) ^ (uint64_t)(uintptr_t)j);
    if (max_tokens < 0) max_tokens = 0;
    if (max_tokens > room) max_tokens = room;
    trace_event(s, trace_id, "prefill done; decode_max=%d ctx_room=%d", max_tokens, room);
    const double decode_t0 = now_sec();
    double last_decode_log_t = decode_t0;
    int last_decode_log_completion = 0;
    thinking_state thinking = thinking_state_from_prompt(&j->req);
    const bool thinking_gates_tool_markers = ds4_think_mode_enabled(j->req.think_mode);
    bool tool_scan_waiting_for_think_close =
        thinking_gates_tool_markers && thinking.inside;
    dsml_decode_tracker dsml_tracker;
    dsml_decode_tracker_init(&dsml_tracker);

    while (!g_stop_requested && completion < max_tokens &&
           ds4_session_pos(s->session) < ds4_session_ctx(s->session)) {
        dsml_decode_state dsml_state = j->req.kind == REQ_CHAT && j->req.has_tools ?
            dsml_tracker.decode : DSML_DECODE_OUTSIDE;
        const bool in_tool_call = dsml_decode_state_is_tool(dsml_state);
        if (!(j->req.kind == REQ_CHAT && j->req.has_tools && (saw_tool_start || in_tool_call))) {
            kv_cache_maybe_store_continued(s);
        }
        float temperature = j->req.temperature;
        int top_k = j->req.top_k;
        float top_p = j->req.top_p;
        float min_p = j->req.min_p;
        if (ds4_think_mode_enabled(j->req.think_mode)) {
            temperature = DS4_DEFAULT_TEMPERATURE;
            top_k = 0;
            top_p = DS4_DEFAULT_TOP_P;
            min_p = DS4_DEFAULT_MIN_P;
        }
        if (in_tool_call && !dsml_decode_state_uses_payload_sampling(dsml_state)) {
            temperature = 0.0f;
        }
        int token = ds4_session_sample(s->session, temperature, top_k, top_p, min_p, &rng);
        if (token == ds4_token_eos(s->engine)) {
            finish = "stop";
            break;
        }

        int toks[17];
        int ntok = 0;
        if (temperature <= 0.0f &&
            ds4_engine_mtp_draft_tokens(s->engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL)
        {
            ntok = ds4_session_eval_speculative_argmax(s->session,
                                                       token,
                                                       max_tokens - completion,
                                                       ds4_token_eos(s->engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                finish = "error";
                break;
            }
        } else {
            if (ds4_session_eval(s->session, token, err, sizeof(err)) != 0) {
                finish = "error";
                break;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop_decode = false;
        for (int ti = 0; ti < ntok && completion < max_tokens; ti++) {
            token = toks[ti];
            if (token == ds4_token_eos(s->engine)) {
                finish = "stop";
                stop_decode = true;
                break;
            }

            size_t piece_len = 0;
            char *piece = ds4_token_text(s->engine, token, &piece_len);
            completion++;

            trace_piece(s, trace_id, piece, piece_len);
            buf_append(&text, piece, piece_len);
            thinking_state_feed(&thinking, piece, piece_len);
            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                dsml_decode_tracker_update(&dsml_tracker, text.ptr, text.len);
            }

            size_t stop_pos = 0, stop_len = 0;
            bool hit_stop = stop_list_find_from(&j->req.stops, text.ptr,
                                                stop_scan_from,
                                                &stop_pos, &stop_len);
            size_t stream_len = hit_stop ?
                stop_pos : stop_list_stream_safe_len(&j->req.stops, text.len);
            if (stream_len > text.len) stream_len = text.len;
            stream_len = utf8_stream_safe_len(text.ptr, plain_stream_pos,
                                              stream_len, hit_stop);
            if (!hit_stop && j->req.stops.max_len > 1) {
                const size_t hold = j->req.stops.max_len - 1;
                stop_scan_from = text.len > hold ? text.len - hold : 0;
            }

            if (j->req.stream && !structured_stream && stream_len > plain_stream_pos) {
                char *delta = xstrndup(text.ptr + plain_stream_pos, stream_len - plain_stream_pos);
                bool ok = sse_chunk(j->fd, &j->req, id, delta, NULL);
                free(delta);
                if (!ok) {
                    finish = "error";
                    snprintf(err, sizeof(err), "client stream write failed");
                    free(piece);
                    stop_decode = true;
                    break;
                }
                plain_stream_pos = stream_len;
            }
            if (j->req.stream && j->req.api == API_ANTHROPIC &&
                !anthropic_sse_stream_update(j->fd, s, &j->req, id,
                                             &anthropic_live, text.ptr, stream_len,
                                             false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (openai_live_chat &&
                !openai_sse_stream_update(j->fd, s, &j->req, id,
                                          &openai_live, text.ptr, stream_len,
                                          false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (responses_live_chat &&
                !responses_sse_stream_update(j->fd, &j->req,
                                             &responses_live, text.ptr, stream_len,
                                             false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            free(piece);

            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                if (thinking_gates_tool_markers && thinking.inside) {
                    /* A DSML block inside reasoning is not executable.  This is
                     * the live guard: do not let a quoted or mistaken marker in
                     * <think> stop decoding as a real tool call. */
                    tool_scan_waiting_for_think_close = true;
                    tool_scan_from = text.len;
                } else {
                    if (tool_scan_waiting_for_think_close) {
                        const char *think_end = find_last_substr(text.ptr, "</think>");
                        tool_scan_from = think_end ? (size_t)((think_end + 8) - text.ptr) : text.len;
                        if (tool_scan_from > text.len) tool_scan_from = text.len;
                        tool_scan_waiting_for_think_close = false;
                    }
                    if (tool_scan_from > text.len) tool_scan_from = text.len;
                    const char *tool_scan = text.ptr ? text.ptr + tool_scan_from : "";
                    bool orphan_end = false;
                    bool old_start = saw_tool_start;
                    bool old_end = saw_tool_end;
                    observe_tool_markers(tool_scan, &saw_tool_start, &saw_tool_end, &orphan_end);
                    if (orphan_end && !saw_orphan_tool_end) {
                        saw_orphan_tool_end = true;
                        server_log(DS4_LOG_WARNING,
                                   "ds4-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens",
                                   ctx_span,
                                   req_flags[0] ? " " : "",
                                   req_flags,
                                   completion);
                        trace_event(s, trace_id,
                                    "ignored orphan tool-call end marker after %d generated tokens",
                                    completion);
                    }
                    if (saw_tool_start && !old_start) {
                        trace_event(s, trace_id, "entered tool-call block after %d generated tokens", completion);
                    }
                    if (saw_tool_end && !old_end) {
                        trace_event(s, trace_id, "closed tool-call block after %d generated tokens", completion);
                    }
                    const size_t marker_hold = 80;
                    size_t hold_from = text.len > marker_hold ? text.len - marker_hold : 0;
                    if (hold_from > tool_scan_from) tool_scan_from = hold_from;
                    if (s->trace && completion >= next_tool_progress) {
                        trace_event(s, trace_id,
                                    "progress gen=%d dsml_start=%d dsml_end=%d",
                                    completion, saw_tool_start ? 1 : 0, saw_tool_end ? 1 : 0);
                        next_tool_progress += 128;
                    }
                }
            }

            if (completion >= next_decode_log) {
                log_decode_progress(j->req.kind, prompt_tokens, completion,
                                    responses_protocol,
                                    j->req.has_tools,
                                    thinking.inside,
                                    saw_tool_start,
                                    saw_tool_end,
                                    decode_t0,
                                    &last_decode_log_t,
                                    &last_decode_log_completion);
                next_decode_log += 50;
            }

            if (hit_stop) {
                (void)stop_len;
                finish = "stop";
                text.len = stop_pos;
                text.ptr[text.len] = '\0';
                ds4_session_invalidate(s->session);
                stop_decode = true;
                break;
            }

            if (j->req.kind == REQ_CHAT && j->req.has_tools && saw_tool_end) {
                finish = "tool_calls";
                stop_decode = true;
                break;
            }
        }
        if (stop_decode) break;
    }

    if (g_stop_requested && strcmp(finish, "error") != 0) {
        finish = "error";
        snprintf(err, sizeof(err), "shutdown requested");
    }

    if (j->req.kind == REQ_CHAT && j->req.has_tools &&
        saw_tool_start && !saw_tool_end && strcmp(finish, "error") != 0)
    {
        /* A partial streamed tool call cannot be retracted.  If the model ends
         * before closing the DSML block, fail the turn instead of letting clients
         * execute an incomplete `{}` or partially parsed argument object. */
        finish = "error";
        snprintf(err, sizeof(err), "unterminated tool call");
    }

    if (completion > last_decode_log_completion) {
        log_decode_progress(j->req.kind, prompt_tokens, completion,
                            responses_protocol,
                            j->req.has_tools,
                            thinking.inside,
                            saw_tool_start,
                            saw_tool_end,
                            decode_t0,
                            &last_decode_log_t,
                            &last_decode_log_completion);
    }

    if (j->req.stream && !structured_stream && text.len > plain_stream_pos) {
        char *tail = xstrndup(text.ptr + plain_stream_pos, text.len - plain_stream_pos);
        if (!sse_chunk(j->fd, &j->req, id, tail, NULL)) finish = "error";
        free(tail);
    }

    tool_calls parsed_calls = {0};
    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    const char *final_finish = finish;
    bool recovered_tool_parse_failure = false;
    if (j->req.kind == REQ_CHAT) {
        bool parsed_ok = parse_generated_message_for_response(
            text.ptr ? text.ptr : "",
            j->req.has_tools,
            saw_tool_start,
            ds4_think_mode_enabled(j->req.think_mode),
            &final_finish,
            err,
            sizeof(err),
            &parsed_content,
            &parsed_reasoning,
            &parsed_calls,
            &recovered_tool_parse_failure);
        if (!parsed_ok && recovered_tool_parse_failure) {
            server_log(DS4_LOG_WARNING,
                       "ds4-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s",
                       ctx_span,
                       req_flags[0] ? " " : "",
                       req_flags,
                       final_finish);
            trace_event(s, trace_id,
                        "invalid tool call returned as assistant text finish=%s",
                        final_finish);
        }
        if (parsed_calls.len) {
            if (openai_live_chat) apply_openai_stream_tool_ids(&parsed_calls, &openai_live);
            if (j->req.api == API_ANTHROPIC && j->req.stream)
                apply_anthropic_stream_tool_ids(&parsed_calls, &anthropic_live);
            assign_tool_call_ids(s, &parsed_calls, j->req.api);
            tool_memory_remember(s, &parsed_calls);
            final_finish = "tool_calls";
        } else if (j->req.api == API_RESPONSES) {
            responses_live_clear(s);
        }
    }
    log_tool_calls_summary(ctx_span, &parsed_calls,
                           responses_protocol);

    trace_finish(s, trace_id, &j->req, final_finish, completion,
                 saw_tool_start, saw_tool_end,
                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                 parsed_reasoning, &parsed_calls, now_sec() - t0);

    char *responses_visible = NULL;
    if (j->req.api == API_RESPONSES) {
        if (strcmp(final_finish, "error")) {
            /* Store the post-turn visible transcript plus the live token
             * frontier.  The next Responses request may replay only this
             * visible surface, while the real session also contains hidden
             * reasoning and exact sampled tool-call bytes. */
            responses_visible =
                responses_build_visible_transcript(&j->req,
                    parsed_content ? parsed_content : "",
                    parsed_reasoning, &parsed_calls);
            responses_live_remember(s, responses_visible ? responses_visible : "",
                                    parsed_calls.len ? &parsed_calls : NULL);
            continuation_store_response(s, &j->req, responses_response_id,
                                        responses_visible ? responses_visible : "");
        } else {
            responses_live_clear(s);
        }
    }
    if (j->req.api == API_ANTHROPIC) {
        if (parsed_calls.len && strcmp(final_finish, "error") &&
            strcmp(final_finish, "length"))
        {
            anthropic_live_remember(s, &parsed_calls);
        } else {
            anthropic_live_clear(s);
        }
    }

    if (j->req.kind == REQ_CHAT && parsed_calls.len &&
        j->req.api != API_RESPONSES &&
        should_canonicalize_tool_checkpoint(s, &parsed_calls))
    {
        /* Chat/completions has no protocol object that binds the next request
         * to this live KV state.  Canonicalize only the fallback tool-call
         * path where we lack exact sampled DSML replay; when raw DSML is known,
         * replaying those bytes keeps future prompts aligned without rebuilding
         * hidden reasoning.  Responses deliberately skips this path because its
         * previous_response_id contract binds the next turn to live state. */
        canonicalize_tool_checkpoint(s, j, ctx_span, trace_id,
                                     parsed_content ? parsed_content : "",
                                     parsed_reasoning, &parsed_calls);
        thinking_live_clear(s);
    } else if (parsed_calls.len) {
        thinking_live_clear(s);
    } else if (!parsed_calls.len &&
               should_remember_thinking_checkpoint(&j->req, &thinking, final_finish)) {
        remember_thinking_checkpoint(s, j, ctx_span, trace_id,
                                     parsed_content ? parsed_content : "");
    } else if (!parsed_calls.len) {
        thinking_live_clear(s);
    }

    if (j->req.stream) {
        bool response_ok = true;
        if (j->req.api == API_ANTHROPIC) {
            response_ok = anthropic_sse_finish_live(j->fd, s, &j->req, id, &anthropic_live,
                                                    text.ptr ? text.ptr : "", text.len,
                                                    &parsed_calls, final_finish, completion);
        } else if (openai_live_chat) {
            response_ok = openai_sse_finish_live(j->fd, s, &j->req, id, &openai_live,
                                                 text.ptr ? text.ptr : "", text.len,
                                                 &parsed_calls, final_finish,
                                                 prompt_tokens, completion);
        } else if (responses_live_chat) {
            /* If parse recovered a malformed tool call back to plain text,
             * pass parsed_content so the streaming tail can be flushed; in
             * the normal path parsed_content is the assistant text we already
             * streamed and the diff is empty. */
            const char *recover =
                recovered_tool_parse_failure ? parsed_content : NULL;
            response_ok = responses_sse_finish_live(j->fd, &j->req, &responses_live,
                                                    text.ptr ? text.ptr : "", text.len,
                                                    recover,
                                                    &parsed_calls, final_finish,
                                                    prompt_tokens, completion,
                                                    responses_created_at);
        } else if (structured_stream) {
            response_ok = sse_chat_finish(j->fd, &j->req, id,
                                          parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                          parsed_reasoning,
                                          &parsed_calls, final_finish,
                                          prompt_tokens, completion);
        } else {
            response_ok = sse_chunk(j->fd, &j->req, id, NULL, final_finish) &&
                          sse_done(j->fd, &j->req, id, prompt_tokens, completion);
        }
        if (!response_ok) {
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: %s ctx=%s%s%s final stream failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       req_flags[0] ? " " : "",
                       req_flags);
        }
    } else if (j->req.api == API_ANTHROPIC) {
        anthropic_final_response(j->fd, s->enable_cors, &j->req, id,
                                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 prompt_tokens, completion);
    } else if (j->req.api == API_RESPONSES) {
        responses_final_response(j->fd, s->enable_cors, &j->req, responses_response_id,
                                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 prompt_tokens, completion);
    } else {
        final_response(j->fd, s->enable_cors, &j->req, id,
                       parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                       parsed_reasoning,
                       &parsed_calls, final_finish,
                       prompt_tokens, completion);
    }
    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  responses_protocol,
                  true,
                  thinking.inside,
                  saw_tool_start,
                  saw_tool_end);
        if (!strcmp(final_finish, "error") && err[0]) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: chat ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       err,
                       now_sec() - t0);
        } else {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: chat ctx=%s gen=%d%s%s finish=%s %.3fs",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       now_sec() - t0);
        }
    } else {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  responses_protocol,
                  j->req.has_tools,
                  thinking.inside,
                  false,
                  false);
        if (!strcmp(final_finish, "error") && err[0]) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       err,
                       now_sec() - t0);
        } else {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       now_sec() - t0);
        }
    }
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&parsed_calls);
    anthropic_stream_free(&anthropic_live);
    openai_stream_free(&openai_live);
    responses_stream_free(&responses_live);
    free(responses_visible);
    buf_free(&text);
    ds4_tokens_free(&effective_prompt);
}
