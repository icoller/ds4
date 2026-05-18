#ifndef DS4_SERVER_DATA_API_H
#define DS4_SERVER_DATA_API_H

#ifndef DS4_SERVER_DATA_LINKAGE
#define DS4_SERVER_DATA_LINKAGE static
#endif

DS4_SERVER_DATA_LINKAGE void tool_call_free(tool_call *tc);
DS4_SERVER_DATA_LINKAGE void tool_calls_free(tool_calls *calls);
DS4_SERVER_DATA_LINKAGE void tool_calls_push(tool_calls *calls, tool_call tc);
DS4_SERVER_DATA_LINKAGE void chat_msg_add_tool_call_id(chat_msg *m, const char *id);
DS4_SERVER_DATA_LINKAGE void chat_msg_free(chat_msg *m);
DS4_SERVER_DATA_LINKAGE void chat_msgs_free(chat_msgs *msgs);
DS4_SERVER_DATA_LINKAGE void chat_msgs_push(chat_msgs *msgs, chat_msg msg);
DS4_SERVER_DATA_LINKAGE tool_call tool_call_clone(const tool_call *src);
DS4_SERVER_DATA_LINKAGE void tool_calls_copy(tool_calls *dst, const tool_calls *src);
DS4_SERVER_DATA_LINKAGE chat_msg chat_msg_clone(const chat_msg *src);
DS4_SERVER_DATA_LINKAGE void tool_schema_order_free(tool_schema_order *o);
DS4_SERVER_DATA_LINKAGE void tool_schema_orders_free(tool_schema_orders *orders);
DS4_SERVER_DATA_LINKAGE void tool_schema_order_prop_push(tool_schema_order *o, char *prop);
DS4_SERVER_DATA_LINKAGE int tool_schema_orders_find_index(const tool_schema_orders *orders,
                                                          const char *name);
DS4_SERVER_DATA_LINKAGE void tool_schema_orders_push(tool_schema_orders *orders,
                                                     tool_schema_order order);
DS4_SERVER_DATA_LINKAGE const tool_schema_order *tool_schema_orders_find(
    const tool_schema_orders *orders, const char *name);

#endif
