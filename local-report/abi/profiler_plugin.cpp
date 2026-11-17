#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../../nccl-tests-master/src/nccl_profiler.h"

struct Context { FILE* output; unsigned int sequence; };
struct EventHandle { Context* context; unsigned int sequence; unsigned int eventId; const char* type; };
static Context context;
static const int eventMask = ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep;

static ncclResult_t init(void** result, uint64_t, int* mask, const char*, int, int, int, ncclDebugLogger_t) {
  const char* path = getenv("NCCL_PROFILER_JSONL");
  if (!path || !*path) return ncclSystemError;
  context.output = fopen(path, "ab");
  if (!context.output) return ncclSystemError;
  context.sequence = 0;
  *result = &context;
  *mask = eventMask;
  return ncclSuccess;
}
static ncclResult_t startEvent(void* opaque, void** result, ncclProfilerEventDescr_v7_t* description) {
  if (!opaque || !description || !result) return ncclInvalidArgument;
  Context* state = static_cast<Context*>(opaque);
  EventHandle* handle = new EventHandle{state, ++state->sequence, state->sequence, nullptr};
  const char* type = description->type == ncclProfileColl ? "collective" :
    description->type == ncclProfileProxyOp ? "proxy_op" :
    description->type == ncclProfileProxyStep ? "proxy_step" : nullptr;
  if (!type) { delete handle; return ncclInvalidArgument; }
  handle->type = type;
  const long long timestamp = static_cast<long long>(GetTickCount64()) * 1000000;
  if (description->type == ncclProfileColl) {
    fprintf(state->output, "{\"schema_version\":\"1.0\",\"run_id\":\"abi-harness\",\"event_id\":\"abi-%u-start\",\"parent_event_id\":null,\"source\":\"abi_harness\",\"source_event_type\":\"%s\",\"rank\":%d,\"communicator_key\":\"comm-1\",\"collective\":\"%s\",\"lifecycle\":\"start\",\"source_clock_domain\":\"host_monotonic\",\"source_timestamp_ns\":%lld}\n",
      handle->eventId, type, description->rank, description->coll.func, timestamp);
  } else {
    const int peer = description->type == ncclProfileProxyOp ? description->proxyOp.peer : -1;
    const int channel = description->type == ncclProfileProxyOp ? description->proxyOp.channelId : -1;
    const int step = description->type == ncclProfileProxyStep ? description->proxyStep.step : -1;
    fprintf(state->output, "{\"schema_version\":\"1.0\",\"run_id\":\"abi-harness\",\"event_id\":\"abi-%u-start\",\"parent_event_id\":\"abi-1-start\",\"source\":\"abi_harness\",\"source_event_type\":\"%s\",\"rank\":%d,\"channel_id\":%d,\"peer\":%d,\"step\":%d,\"lifecycle\":\"start\",\"source_clock_domain\":\"host_monotonic\",\"source_timestamp_ns\":%lld}\n",
      handle->eventId, type, description->rank, channel, peer, step, timestamp);
  }
  fflush(state->output);
  *result = handle;
  return ncclSuccess;
}
static ncclResult_t stopEvent(void* opaque) {
  if (!opaque) return ncclInvalidArgument;
  EventHandle* handle = static_cast<EventHandle*>(opaque);
  const long long timestamp = static_cast<long long>(GetTickCount64()) * 1000000;
  fprintf(handle->context->output, "{\"schema_version\":\"1.0\",\"run_id\":\"abi-harness\",\"event_id\":\"abi-%u-stop\",\"parent_event_id\":\"abi-%u-start\",\"source\":\"abi_harness\",\"source_event_type\":\"%s\",\"rank\":0,\"lifecycle\":\"stop\",\"source_clock_domain\":\"host_monotonic\",\"source_timestamp_ns\":%lld}\n",
    handle->eventId, handle->eventId, handle->type, timestamp);
  fflush(handle->context->output);
  delete handle;
  return ncclSuccess;
}
static ncclResult_t recordEventState(void* opaque, ncclProfilerEventState_v7_t, ncclProfilerEventStateArgs_v7_t*) {
  return opaque ? ncclSuccess : ncclInvalidArgument;
}
static ncclResult_t finalize(void* opaque) {
  if (opaque != &context || !context.output) return ncclInvalidArgument;
  const int result = fclose(context.output);
  context.output = nullptr;
  return result == 0 ? ncclSuccess : ncclSystemError;
}
static ncclProfiler_v7_t profiler = {"abi-harness-profiler", init, startEvent, stopEvent, recordEventState, finalize};

extern "C" __declspec(dllexport) ncclProfiler_v7_t* ncclProfiler_v7() { return &profiler; }
