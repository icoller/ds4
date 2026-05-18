DS4_SERVER_DATA_LINKAGE void tool_call_free(tool_call *tc) {
    free(tc->id);
    free(tc->name);
    free(tc->arguments);
    memset(tc, 0, sizeof(*tc));
}

DS4_SERVER_DATA_LINKAGE void tool_calls_free(tool_calls *calls) {
    for (int i = 0; i < calls->len; i++) tool_call_free(&calls->v[i]);
    free(calls->raw_dsml);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}

DS4_SERVER_DATA_LINKAGE void tool_calls_push(tool_calls *calls, tool_call tc) {
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 4;
        calls->v = xrealloc(calls->v, (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = tc;
}

DS4_SERVER_DATA_LINKAGE void chat_msg_add_tool_call_id(chat_msg *m, const char *id) {
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

DS4_SERVER_DATA_LINKAGE void chat_msg_free(chat_msg *m) {
    free(m->role);
    free(m->content);
    free(m->reasoning);
    free(m->tool_call_id);
    for (int i = 0; i < m->tool_call_ids_len; i++) free(m->tool_call_ids[i]);
    free(m->tool_call_ids);
    tool_calls_free(&m->calls);
    memset(m, 0, sizeof(*m));
}

DS4_SERVER_DATA_LINKAGE void chat_msgs_free(chat_msgs *msgs) {
    for (int i = 0; i < msgs->len; i++) chat_msg_free(&msgs->v[i]);
    free(msgs->v);
    memset(msgs, 0, sizeof(*msgs));
}

DS4_SERVER_DATA_LINKAGE void chat_msgs_push(chat_msgs *msgs, chat_msg msg) {
    if (msgs->len == msgs->cap) {
        msgs->cap = msgs->cap ? msgs->cap * 2 : 8;
        msgs->v = xrealloc(msgs->v, (size_t)msgs->cap * sizeof(msgs->v[0]));
    }
    msgs->v[msgs->len++] = msg;
}

DS4_SERVER_DATA_LINKAGE tool_call tool_call_clone(const tool_call *src) {
    tool_call dst = {0};
    if (!src) return dst;
    dst.id = src->id ? xstrdup(src->id) : NULL;
    dst.name = src->name ? xstrdup(src->name) : NULL;
    dst.arguments = src->arguments ? xstrdup(src->arguments) : NULL;
    return dst;
}

DS4_SERVER_DATA_LINKAGE void tool_calls_copy(tool_calls *dst, const tool_calls *src) {
    memset(dst, 0, sizeof(*dst));
    if (!src) return;
    dst->raw_dsml = src->raw_dsml ? xstrdup(src->raw_dsml) : NULL;
    for (int i = 0; i < src->len; i++) {
        tool_calls_push(dst, tool_call_clone(&src->v[i]));
    }
}

DS4_SERVER_DATA_LINKAGE chat_msg chat_msg_clone(const chat_msg *src) {
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

DS4_SERVER_DATA_LINKAGE void tool_schema_order_free(tool_schema_order *o) {
    free(o->name);
    free(o->wire_name);
    free(o->namespace);
    for (int i = 0; i < o->len; i++) free(o->prop[i]);
    free(o->prop);
    memset(o, 0, sizeof(*o));
}

DS4_SERVER_DATA_LINKAGE void tool_schema_orders_free(tool_schema_orders *orders) {
    for (int i = 0; i < orders->len; i++) tool_schema_order_free(&orders->v[i]);
    free(orders->v);
    memset(orders, 0, sizeof(*orders));
}

DS4_SERVER_DATA_LINKAGE void tool_schema_order_prop_push(tool_schema_order *o, char *prop) {
    if (o->len == o->cap) {
        o->cap = o->cap ? o->cap * 2 : 8;
        o->prop = xrealloc(o->prop, (size_t)o->cap * sizeof(o->prop[0]));
    }
    o->prop[o->len++] = prop;
}

DS4_SERVER_DATA_LINKAGE int tool_schema_orders_find_index(const tool_schema_orders *orders,
                                                          const char *name) {
    if (!orders || !name) return -1;
    for (int i = 0; i < orders->len; i++) {
        if (orders->v[i].name && !strcmp(orders->v[i].name, name)) return i;
    }
    return -1;
}

DS4_SERVER_DATA_LINKAGE void tool_schema_orders_push(tool_schema_orders *orders,
                                                     tool_schema_order order) {
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

DS4_SERVER_DATA_LINKAGE const tool_schema_order *tool_schema_orders_find(
    const tool_schema_orders *orders, const char *name) {
    int idx = tool_schema_orders_find_index(orders, name);
    return idx >= 0 ? &orders->v[idx] : NULL;
}
