#include "ds4.h"
#include "rax.h"

/* OpenAI/Anthropic compatible local server.
 *
 * HTTP is intentionally simple: each client connection is handled by a small
 * blocking thread that parses one request, then queues a job to the single
 * Metal worker.  The worker owns the ds4_session and therefore owns all live KV
 * cache state.  That keeps session reuse, disk checkpointing, and future
 * batching decisions in one place instead of spreading graph mutations across
 * client threads. */

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop_requested = 0;
static volatile sig_atomic_t g_listen_fd = -1;

#define DS4_SERVER_IO_TIMEOUT_SEC 10
#define DS4_SERVER_SEND_STALL_TIMEOUT_MS 2000

static void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested) _exit(130);
    g_stop_requested = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}

#include "ds4_server_base.inc"

#include "ds4_server_types.h"

static void tool_call_free(tool_call *tc) {
    free(tc->id);
    free(tc->name);
    free(tc->arguments);
    memset(tc, 0, sizeof(*tc));
}

static void tool_calls_free(tool_calls *calls) {
    for (int i = 0; i < calls->len; i++) tool_call_free(&calls->v[i]);
    free(calls->raw_dsml);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}

static void tool_calls_push(tool_calls *calls, tool_call tc) {
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 4;
        calls->v = xrealloc(calls->v, (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = tc;
}

static void chat_msg_add_tool_call_id(chat_msg *m, const char *id) {
    if (!m || !id || !id[0]) return;
    if (!m->tool_call_id) m->tool_call_id = xstrdup(id);
    for (int i = 0; i < m->tool_call_ids_len; i++) {
        if (m->tool_call_ids[i] && !strcmp(m->tool_call_ids[i], id)) return;
    }
    if (m->tool_call_ids_len == m->tool_call_ids_cap) {
        m->tool_call_ids_cap = m->tool_call_ids_cap ? m->tool_call_ids_cap * 2 : 2;
        m->tool_call_ids = xrealloc(m->tool_call_ids,
            (size_t)m->tool_call_ids_cap * sizeof(m->tool_call_ids[0]));
    }
    m->tool_call_ids[m->tool_call_ids_len++] = xstrdup(id);
}

static void chat_msg_free(chat_msg *m) {
    free(m->role);
    free(m->content);
    free(m->reasoning);
    free(m->tool_call_id);
    for (int i = 0; i < m->tool_call_ids_len; i++) free(m->tool_call_ids[i]);
    free(m->tool_call_ids);
    tool_calls_free(&m->calls);
    memset(m, 0, sizeof(*m));
}

static void chat_msgs_free(chat_msgs *msgs) {
    for (int i = 0; i < msgs->len; i++) chat_msg_free(&msgs->v[i]);
    free(msgs->v);
    memset(msgs, 0, sizeof(*msgs));
}

static void chat_msgs_push(chat_msgs *msgs, chat_msg msg) {
    if (msgs->len == msgs->cap) {
        msgs->cap = msgs->cap ? msgs->cap * 2 : 8;
        msgs->v = xrealloc(msgs->v, (size_t)msgs->cap * sizeof(msgs->v[0]));
    }
    msgs->v[msgs->len++] = msg;
}

static tool_call tool_call_clone(const tool_call *src) {
    tool_call dst = {0};
    if (!src) return dst;
    dst.id = src->id ? xstrdup(src->id) : NULL;
    dst.name = src->name ? xstrdup(src->name) : NULL;
    dst.arguments = src->arguments ? xstrdup(src->arguments) : NULL;
    return dst;
}

static void tool_calls_copy(tool_calls *dst, const tool_calls *src) {
    memset(dst, 0, sizeof(*dst));
    if (!src) return;
    dst->raw_dsml = src->raw_dsml ? xstrdup(src->raw_dsml) : NULL;
    for (int i = 0; i < src->len; i++) {
        tool_calls_push(dst, tool_call_clone(&src->v[i]));
    }
}

static chat_msg chat_msg_clone(const chat_msg *src) {
    chat_msg dst = {0};
    if (!src) return dst;
    dst.role = src->role ? xstrdup(src->role) : NULL;
    dst.content = src->content ? xstrdup(src->content) : NULL;
    dst.reasoning = src->reasoning ? xstrdup(src->reasoning) : NULL;
    dst.tool_call_id = src->tool_call_id ? xstrdup(src->tool_call_id) : NULL;
    for (int i = 0; i < src->tool_call_ids_len; i++) {
        chat_msg_add_tool_call_id(&dst, src->tool_call_ids[i]);
    }
    tool_calls_copy(&dst.calls, &src->calls);
    return dst;
}

static void tool_schema_order_free(tool_schema_order *o) {
    free(o->name);
    free(o->wire_name);
    free(o->namespace);
    for (int i = 0; i < o->len; i++) free(o->prop[i]);
    free(o->prop);
    memset(o, 0, sizeof(*o));
}

static void tool_schema_orders_free(tool_schema_orders *orders) {
    for (int i = 0; i < orders->len; i++) tool_schema_order_free(&orders->v[i]);
    free(orders->v);
    memset(orders, 0, sizeof(*orders));
}

static void tool_schema_order_prop_push(tool_schema_order *o, char *prop) {
    if (o->len == o->cap) {
        o->cap = o->cap ? o->cap * 2 : 8;
        o->prop = xrealloc(o->prop, (size_t)o->cap * sizeof(o->prop[0]));
    }
    o->prop[o->len++] = prop;
}

static int tool_schema_orders_find_index(const tool_schema_orders *orders, const char *name) {
    if (!orders || !name) return -1;
    for (int i = 0; i < orders->len; i++) {
        if (orders->v[i].name && !strcmp(orders->v[i].name, name)) return i;
    }
    return -1;
}

static void tool_schema_orders_push(tool_schema_orders *orders, tool_schema_order order) {
    int idx = tool_schema_orders_find_index(orders, order.name);
    if (idx >= 0) {
        tool_schema_order_free(&orders->v[idx]);
        orders->v[idx] = order;
        return;
    }
    if (orders->len == orders->cap) {
        orders->cap = orders->cap ? orders->cap * 2 : 8;
        orders->v = xrealloc(orders->v, (size_t)orders->cap * sizeof(orders->v[0]));
    }
    orders->v[orders->len++] = order;
}

static const tool_schema_order *tool_schema_orders_find(const tool_schema_orders *orders, const char *name) {
    int idx = tool_schema_orders_find_index(orders, name);
    return idx >= 0 ? &orders->v[idx] : NULL;
}

#include "ds4_server_parse.inc"

#include "ds4_server_stream.inc"

#include "ds4_server_tool_memory.inc"

#include "ds4_server_kv_cache.inc"

#include "ds4_server_trace.inc"

#include "ds4_server_checkpoint.inc"

#include "ds4_server_generate.inc"

#include "ds4_server_http.inc"

#include "ds4_server_runtime.inc"

#ifndef DS4_SERVER_TEST
int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = stop_signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    server_config cfg = parse_options(argc, argv);
    if (cfg.chdir_path && chdir(cfg.chdir_path) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to chdir to %s: %s",
                   cfg.chdir_path, strerror(errno));
        return 1;
    }

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) return 1;

    log_context_memory(cfg.engine.backend, cfg.ctx_size);

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to create %s session",
                   ds4_backend_name(cfg.engine.backend));
        ds4_engine_close(engine);
        return 1;
    }

    server s;
    memset(&s, 0, sizeof(s));
    s.engine = engine;
    s.session = session;
    s.default_tokens = cfg.default_tokens;
    s.disable_exact_dsml_tool_replay = cfg.disable_exact_dsml_tool_replay;
    s.tool_mem.max_entries = cfg.tool_memory_max_ids;
    s.enable_cors = cfg.enable_cors;
    if (cfg.kv_disk_dir) {
        kv_cache_open(&s.kv, cfg.kv_disk_dir, cfg.kv_disk_space_mb,
                      cfg.kv_cache_reject_different_quant, cfg.kv_cache);
    }
    if (s.disable_exact_dsml_tool_replay) {
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: exact DSML tool replay disabled; tool history uses canonical JSON rendering");
    }
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);
    pthread_cond_init(&s.clients_cv, NULL);
    pthread_mutex_init(&s.tool_mu, NULL);
    pthread_mutex_init(&s.trace_mu, NULL);
    if (cfg.trace_path) {
        s.trace = fopen(cfg.trace_path, "w");
        if (!s.trace) {
            server_log(DS4_LOG_DEFAULT, "ds4-server: failed to open trace file %s: %s",
                       cfg.trace_path, strerror(errno));
            server_close_resources(&s);
            return 1;
        }
        setvbuf(s.trace, NULL, _IONBF, 0);
        server_log(DS4_LOG_DEFAULT, "ds4-server: tracing session to %s", cfg.trace_path);
    }

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_main, &s) != 0) die("failed to start worker");

    int lfd = listen_on(cfg.host, cfg.port);
    if (lfd < 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to listen on %s:%d: %s", cfg.host, cfg.port, strerror(errno));
        pthread_mutex_lock(&s.mu);
        s.stopping = true;
        pthread_cond_broadcast(&s.cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(worker, NULL);
        server_close_resources(&s);
        return 1;
    }
    g_listen_fd = lfd;
    server_log(DS4_LOG_DEFAULT, "ds4-server: listening on http://%s:%d", cfg.host, cfg.port);

    while (!g_stop_requested) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (g_stop_requested) break;
            if (errno == EINTR) continue;
            server_log(DS4_LOG_DEFAULT, "ds4-server: accept failed: %s", strerror(errno));
            continue;
        }
        if (g_stop_requested) {
            close(fd);
            break;
        }

        configure_client_socket(fd);
        client_arg *ca = xmalloc(sizeof(*ca));
        ca->srv = &s;
        ca->fd = fd;
        pthread_mutex_lock(&s.mu);
        s.clients++;
        pthread_mutex_unlock(&s.mu);
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, ca) != 0) {
            pthread_mutex_lock(&s.mu);
            s.clients--;
            pthread_cond_broadcast(&s.clients_cv);
            pthread_mutex_unlock(&s.mu);
            free(ca);
            close(fd);
            continue;
        }
        pthread_detach(th);
    }
    if (g_listen_fd >= 0) {
        close(lfd);
        g_listen_fd = -1;
    }

    server_log(DS4_LOG_DEFAULT, "ds4-server: shutdown requested, draining requests");
    pthread_mutex_lock(&s.mu);
    s.stopping = true;
    pthread_cond_broadcast(&s.cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(worker, NULL);
    pthread_mutex_lock(&s.mu);
    while (s.clients > 0) pthread_cond_wait(&s.clients_cv, &s.mu);
    pthread_mutex_unlock(&s.mu);

    const ds4_tokens *tokens = ds4_session_tokens(s.session);
    if (s.kv.enabled && tokens && tokens->len >= s.kv.opt.min_tokens) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: persisting current KV cache before shutdown tokens=%d",
                   tokens->len);
        kv_cache_store_current(&s, "shutdown");
    }
    server_close_resources(&s);
    return 0;
}
#else
#include "ds4_server_tests.inc"
#endif
