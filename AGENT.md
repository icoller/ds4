# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy, and the production server `main`.
- `ds4_server_types.h`: shared internal server data structures and forward
  declarations for requests, chat/tool payloads, KV metadata, tool memory, and
  `server`/`job`; extracted first in the second-stage modularization to shrink
  `ds4_server.c` without changing linkage.
- `ds4_server_parse.inc`: request parsing, tool schema normalization,
  chat/responses prompt rendering, stop handling, and Responses history trim;
  extracted as one contiguous block to remove the bulk of protocol parsing from
  `ds4_server.c` while keeping the same single translation unit semantics.
- `ds4_server_stream.inc`: HTTP response helpers, SSE framing, DSML stream
  projection, and OpenAI/Responses/Anthropic live streaming state machines;
  extracted as one contiguous block so `ds4_server.c` can collapse to mostly
  wiring and top-level includes.
- `ds4_server_base.inc`: low-level server-only helpers such as dynamic buffers,
  permissive JSON parsing helpers, and request/API enums; kept as a textual
  include during the no-regression split so static linkage and compile behavior
  stay unchanged.
- `ds4_server_trace.inc`: trace diagnostics, prefill/decode progress helpers,
  and related logging utilities; extracted as a textual include before any
  later move to separate translation units.
- `ds4_server_tool_memory.inc`: exact DSML tool-call memory, live continuation
  state, and protocol tool-id reuse helpers; kept as a textual include during
  the fast no-regression split.
- `ds4_server_kv_cache.inc`: disk KV cache metadata, persistence, restore,
  eviction, and visible-prefix checkpoint helpers; extracted as a textual
  include during the no-regression split so behavior and static reachability
  stay unchanged.
- `ds4_server_checkpoint.inc`: checkpoint canonicalization, visible replay
  suffix construction, and continuation/prefill helpers that prepare
  `generate_job()` without changing the worker execution path.
- `ds4_server_generate.inc`: the full `generate_job()` execution path,
  including cache selection, prefill, decode loop, tool-call parsing, live
  continuation state updates, and final response emission; moved intact as a
  textual include before any semantic refactor.
- `ds4_server_http.inc`: worker queue, HTTP parsing, model endpoints, client
  connection handling, and socket/listen helpers; extracted as a textual
  include so the server loop wiring can move without changing behavior.
- `ds4_server_runtime.inc`: server config parsing, usage/help text, backend
  defaults, resource teardown, and other runtime helpers used by the
  production `main`.
- `ds4_server_tests.inc`: server unit/regression tests that are textually
  included only under `DS4_SERVER_TEST` to keep test-only code out of the
  production build while preserving internal static symbol coverage.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.

## Server Split Rules

- Treat `ds4_server.c` splits as no-regression moves first: prefer extracting
  cohesive blocks into root-level `ds4_server_*.inc` files and include them back
  into `ds4_server.c` before introducing new headers or translation units.
- After every server split step, rebuild `ds4_server.o` and the `ds4_test`
  target before proceeding further.
- Keep include files at the repository root during the first split pass; move to
  subdirectories only after boundaries stabilize and build/test wiring is
  already explicit.
