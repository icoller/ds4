#ifndef DS4_SERVER_BASE_TYPES_H
#define DS4_SERVER_BASE_TYPES_H

typedef enum {
    REQ_CHAT,
    REQ_COMPLETION,
} req_kind;

typedef enum {
    API_OPENAI,
    API_ANTHROPIC,
    API_RESPONSES,
} api_style;

#endif
