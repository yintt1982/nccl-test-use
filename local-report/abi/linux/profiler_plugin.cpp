#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "../../../nccl-tests-master/src/nccl_profiler.h"

static const size_t queueCapacity = 4096;
static const size_t lineCapacity = 2048;
struct Context {
  FILE* output;
  unsigned int sequence;
  std::array<std::array<char, lineCapacity>, queueCapacity> queue;
  size_t head;
  size_t tail;
  size_t count;
  uint64_t lost;
  bool stopping;
  bool writerFailed;
  std::mutex mutex;
  std::condition_variable ready;
  std::thread writer;
};
struct EventHandle { Context* context; unsigned int eventId; const char* type; };
static Context context;
static const int eventMask = ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep;

static uint64_t monotonicNs() {
  timespec value{};
  clock_gettime(CLOCK_MONOTONIC, &value);
  return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}
static void writerLoop(Context* state) {
  for (;;) {
    std::array<char, lineCapacity> line;
    {
      std::unique_lock<std::mutex> lock(state->mutex);
      state->ready.wait(lock, [state] { return state->count || state->stopping; });
      if (!state->count && state->stopping) break;
      line = state->queue[state->head];
      state->head = (state->head + 1) % queueCapacity;
      --state->count;
    }
    if (!state->writerFailed && fputs(line.data(), state->output) == EOF) state->writerFailed = true;
  }
  if (fflush(state->output) != 0) state->writerFailed = true;
}
static ncclResult_t enqueue(Context* state, const char* line) {
  std::lock_guard<std::mutex> lock(state->mutex);
  if (state->count == queueCapacity || state->stopping) {
    ++state->lost;
    return ncclInProgress;
  }
  snprintf(state->queue[state->tail].data(), lineCapacity, "%s", line);
  state->tail = (state->tail + 1) % queueCapacity;
  ++state->count;
  state->ready.notify_one();
  return ncclSuccess;
}
static ncclResult_t init(void** result, uint64_t, int* mask, const char*, int, int, int, ncclDebugLogger_t) {
  const char* path = getenv("NCCL_PROFILER_JSONL");
  if (!path || !*path) return ncclSystemError;
  context.output = fopen(path, "ab");
  if (!context.output) return ncclSystemError;
  context.sequence = 0;
  context.head = context.tail = context.count = 0;
  context.lost = 0;
  context.stopping = false;
  context.writerFailed = false;
  context.writer = std::thread(writerLoop, &context);
  *result = &context;
  *mask = eventMask;
  return ncclSuccess;
}
static ncclResult_t startEvent(void* opaque, void** result, ncclProfilerEventDescr_v7_t* description) {
  if (!opaque || !description || !result) return ncclInvalidArgument;
  Context* state = static_cast<Context*>(opaque);
  const char* type = description->type == ncclProfileColl ? "collective" :
    description->type == ncclProfileProxyOp ? "proxy_op" :
    description->type == ncclProfileProxyStep ? "proxy_step" : nullptr;
  if (!type) return ncclInvalidArgument;
  EventHandle* handle = new EventHandle{state, __atomic_add_fetch(&state->sequence, 1, __ATOMIC_RELAXED), type};
  const uint64_t timestamp = monotonicNs();
  char line[lineCapacity];
  if (description->type == ncclProfileColl) {
    snprintf(line, sizeof(line), "{\"schema_version\":\"1.0\",\"run_id\":\"abi-harness\",\"event_id\":\"abi-%u-start\",\"parent_event_id\":null,\"source\":\"abi_harness\",\"source_event_type\":\"%s\",\"rank\":%d,\"communicator_key\":\"comm-1\",\"collective\":\"%s\",\"lifecycle\":\"start\",\"source_clock_domain\":\"host_monotonic\",\"source_timestamp_ns\":%llu}\n",
      handle->eventId, type, description->rank, description->coll.func ? description->coll.func : "unknown", static_cast<unsigned long long>(timestamp));
  } else {
    const int peer = description->type == ncclProfileProxyOp ? description->proxyOp.peer : -1;
    const int channel = description->type == ncclProfileProxyOp ? description->proxyOp.channelId : -1;
    const int step = description->type == ncclProfileProxyStep ? description->proxyStep.step : -1;
    const int isSend = description->type == ncclProfileProxyOp ? description->proxyOp.isSend : -1;
    const int nSteps = description->type == ncclProfileProxyOp ? description->proxyOp.nSteps : -1;
    const int chunkSize = description->type == ncclProfileProxyOp ? description->proxyOp.chunkSize : -1;
    snprintf(line, sizeof(line), "{\"schema_version\":\"1.0\",\"run_id\":\"abi-harness\",\"event_id\":\"abi-%u-start\",\"parent_event_id\":\"abi-1-start\",\"source\":\"abi_harness\",\"source_event_type\":\"%s\",\"rank\":%d,\"channel_id\":%d,\"peer\":%d,\"step\":%d,\"is_send\":%d,\"declared_n_steps\":%d,\"declared_chunk_size_bytes\":%d,\"lifecycle\":\"start\",\"source_clock_domain\":\"host_monotonic\",\"source_timestamp_ns\":%llu}\n",
      handle->eventId, type, description->rank, channel, peer, step, isSend, nSteps, chunkSize, static_cast<unsigned long long>(timestamp));
  }
  *result = handle;
  return enqueue(state, line);
}
static ncclResult_t stopEvent(void* opaque) {
  if (!opaque) return ncclInvalidArgument;
  EventHandle* handle = static_cast<EventHandle*>(opaque);
  char line[lineCapacity];
  snprintf(line, sizeof(line), "{\"schema_version\":\"1.0\",\"run_id\":\"abi-harness\",\"event_id\":\"abi-%u-stop\",\"parent_event_id\":\"abi-%u-start\",\"source\":\"abi_harness\",\"source_event_type\":\"%s\",\"rank\":0,\"lifecycle\":\"stop\",\"source_clock_domain\":\"host_monotonic\",\"source_timestamp_ns\":%llu}\n",
    handle->eventId, handle->eventId, handle->type, static_cast<unsigned long long>(monotonicNs()));
  const ncclResult_t result = enqueue(handle->context, line);
  delete handle;
  return result;
}
static ncclResult_t recordEventState(void* opaque, ncclProfilerEventState_v7_t, ncclProfilerEventStateArgs_v7_t*) {
  return opaque ? ncclSuccess : ncclInvalidArgument;
}
static ncclResult_t finalize(void* opaque) {
  if (opaque != &context || !context.output) return ncclInvalidArgument;
  {
    std::lock_guard<std::mutex> lock(context.mutex);
    context.stopping = true;
  }
  context.ready.notify_one();
  context.writer.join();
  const bool failed = context.writerFailed || context.lost != 0;
  const int result = fclose(context.output);
  context.output = nullptr;
  return result == 0 && !failed ? ncclSuccess : ncclSystemError;
}
static ncclProfiler_v7_t profiler = {"abi-harness-profiler", init, startEvent, stopEvent, recordEventState, finalize};

extern "C" ncclProfiler_v7_t* ncclProfiler_v7() { return &profiler; }
