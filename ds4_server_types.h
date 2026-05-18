#ifndef DS4_SERVER_TYPES_H
#define DS4_SERVER_TYPES_H

#include "ds4_server_base_types.h"

typedef struct server server;
typedef struct job job;

typedef struct {
    char *id;
    char *name;
    char *arguments;
} tool_call;

typedef struct {
    tool_call *v;
    int len;
    int cap;
    char *raw_dsml;
} tool_calls;

typedef struct {
    int mem;
    int disk;
    int canonical;
    int missing_ids;
} tool_replay_stats;

typedef struct {
    char *name;
    char *wire_name;
    char *namespace;
    bool responses_tool_search;
    char **prop;
    int len;
    int cap;
} tool_schema_order;

typedef struct {
    tool_schema_order *v;
    int len;
    int cap;
} tool_schema_orders;

typedef struct {
    char *role;
    char *content;
    char *reasoning;
    char *tool_call_id;
    char **tool_call_ids;
    int tool_call_ids_len;
    int tool_call_ids_cap;
    tool_calls calls;
} chat_msg;

typedef struct {
    chat_msg *v;
    int len;
    int cap;
} chat_msgs;

typedef struct {
    char **v;
    int len;
    int cap;
    size_t max_len;
} stop_list;

typedef struct {
    req_kind kind;
    api_style api;
    ds4_tokens prompt;
    char *model;
    stop_list stops;
    char *raw_body;
    char *prompt_text;
    tool_schema_orders tool_orders;
    int max_tokens;
    int top_k;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    bool stream;
    bool stream_include_usage;
    int cache_read_tokens;
    int cache_write_tokens;
    ds4_think_mode think_mode;
    bool has_tools;
    bool prompt_preserves_reasoning;
    bool reasoning_summary_emit;
    bool responses_requires_live_tool_state;
    bool responses_requires_live_reasoning;
    stop_list responses_live_call_ids;
    char *responses_live_suffix_text;
    bool anthropic_requires_live_tool_state;
    stop_list anthropic_live_call_ids;
    char *anthropic_live_suffix_text;
    bool chat_requires_live_tool_state;
    stop_list chat_live_call_ids;
    char *chat_live_suffix_text;
    char *previous_response_id;
    char *continuation_suffix_text;
    char *client_session_key;
    char *implicit_session_basis;
    char *implicit_session_key;
    bool prompt_anatomy_valid;
    int prompt_system_msgs;
    int prompt_user_msgs;
    int prompt_assistant_msgs;
    int prompt_tool_msgs;
    int prompt_tool_schema_tokens_approx;
    int prompt_system_tokens_approx;
    int prompt_recent_tail_tokens_approx;
    tool_replay_stats tool_replay;
} request;

static void tool_memory_attach_to_messages(server *s, chat_msgs *msgs,
                                           tool_replay_stats *stats);
static bool tool_memory_has_id(server *s, const char *id);
static void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs);
static bool responses_trim_history_to_context(ds4_engine *e, chat_msgs *msgs,
                                              const char *tool_schemas,
                                              const tool_schema_orders *orders,
                                              ds4_think_mode think_mode,
                                              int ctx_size);
static void stop_list_clear(stop_list *stops);
static bool id_list_contains(const stop_list *ids, const char *id);
static void id_list_push_unique(stop_list *ids, const char *id);
static void id_list_free(stop_list *ids);
static bool chat_live_has_call_id(server *s, const char *id);
static bool responses_live_has_call_id(server *s, const char *id);
static bool anthropic_live_has_call_id(server *s, const char *id);

typedef struct {
    char sha[41];
    char *path;
    uint8_t quant_bits;
    uint8_t reason;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint8_t ext_flags;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t text_bytes;
    uint64_t file_size;
} kv_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} kv_cache_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    kv_cache_options opt;
    int continued_last_store_tokens;
    kv_entry *entry;
    int len;
    int cap;
} kv_disk_cache;

typedef enum {
    TOOL_MEMORY_RAM = 0,
    TOOL_MEMORY_DISK = 1,
} tool_memory_source;

typedef struct tool_memory_entry tool_memory_entry;

typedef struct {
    char *dsml;
    size_t len;
    size_t bytes;
    int refs;
    uint64_t seen;
    tool_memory_entry *entries;
} tool_memory_block;

struct tool_memory_entry {
    char *id;
    tool_memory_block *block;
    size_t bytes;
    uint64_t stamp;
    tool_memory_source source;
    tool_memory_entry *prev;
    tool_memory_entry *next;
    tool_memory_entry *block_next;
};

typedef struct {
    rax *by_id;
    rax *by_block;
    tool_memory_entry *head;
    tool_memory_entry *tail;
    int entries;
    int max_entries;
    size_t bytes;
    size_t max_bytes;
    uint64_t clock;
    uint64_t scan_clock;
} tool_memory;

typedef struct {
    bool valid;
    int live_tokens;
    char *visible_text;
    size_t visible_len;
    stop_list call_ids;
} live_tool_state;

typedef struct {
    bool valid;
    int live_tokens;
    char *visible_text;
    size_t visible_len;
} visible_live_state;

typedef struct continuation_entry continuation_entry;

struct continuation_entry {
    char *response_id;
    char *visible_text;
    char *model;
    char *implicit_session_key;
    ds4_session_snapshot snapshot;
    uint64_t created_at;
    uint64_t last_used;
    continuation_entry *prev;
    continuation_entry *next;
};

struct server {
    ds4_engine *engine;
    ds4_session *session;
    int default_tokens;
    kv_disk_cache kv;
    tool_memory tool_mem;
    live_tool_state chat_live;
    live_tool_state responses_live;
    live_tool_state anthropic_live;
    visible_live_state replay_live;
    visible_live_state thinking_live;
    bool disable_exact_dsml_tool_replay;
    bool enable_cors;
    pthread_mutex_t tool_mu;
    pthread_mutex_t continuation_mu;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t clients_cv;
    job *head;
    job *tail;
    bool stopping;
    int clients;
    uint64_t seq;
    FILE *trace;
    pthread_mutex_t trace_mu;
    uint64_t trace_seq;
    rax *continuations_by_id;
    rax *continuation_alias_by_key;
    continuation_entry *continuation_head;
    continuation_entry *continuation_tail;
    int continuation_entries;
};

struct job {
    int fd;
    request req;
    bool done;
    bool stream_headers_sent;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job *next;
};

#endif
