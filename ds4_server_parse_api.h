#ifndef DS4_SERVER_PARSE_API_H
#define DS4_SERVER_PARSE_API_H

#ifndef DS4_SERVER_PARSE_LINKAGE
#define DS4_SERVER_PARSE_LINKAGE static
#endif

DS4_SERVER_PARSE_LINKAGE void request_init(request *r, req_kind kind, int max_tokens);
DS4_SERVER_PARSE_LINKAGE void request_free(request *r);
DS4_SERVER_PARSE_LINKAGE bool parse_reasoning_effort_value(const char **p, ds4_think_mode *out);
DS4_SERVER_PARSE_LINKAGE bool parse_stop(const char **p, stop_list *out);
DS4_SERVER_PARSE_LINKAGE char *render_chat_prompt_text(const chat_msgs *msgs,
                                                       const char *tool_schemas,
                                                       const tool_schema_orders *tool_orders,
                                                       ds4_think_mode think_mode);
DS4_SERVER_PARSE_LINKAGE bool responses_trim_history_to_context(ds4_engine *e,
                                                                chat_msgs *msgs,
                                                                const char *tool_schemas,
                                                                const tool_schema_orders *orders,
                                                                ds4_think_mode think_mode,
                                                                int ctx_size);
DS4_SERVER_PARSE_LINKAGE bool parse_chat_request(ds4_engine *e, server *s,
                                                 const char *body, int def_tokens,
                                                 int ctx_size, request *r,
                                                 char *err, size_t errlen);
DS4_SERVER_PARSE_LINKAGE bool parse_anthropic_request(ds4_engine *e, server *s,
                                                      const char *body, int def_tokens,
                                                      int ctx_size, request *r,
                                                      char *err, size_t errlen);
DS4_SERVER_PARSE_LINKAGE bool parse_responses_request(ds4_engine *e, server *s,
                                                      const char *body, int def_tokens,
                                                      int ctx_size, request *r,
                                                      char *err, size_t errlen);
DS4_SERVER_PARSE_LINKAGE bool parse_completion_request(ds4_engine *e,
                                                       const char *body, int def_tokens,
                                                       int ctx_size, request *r,
                                                       char *err, size_t errlen);

#endif
