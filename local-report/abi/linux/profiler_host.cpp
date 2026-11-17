#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../nccl-tests-master/src/nccl_profiler.h"

typedef ncclProfiler_v7_t* (*GetProfiler)();
static void logger(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
static void require(bool condition, const char* message) {
  if (!condition) { fprintf(stderr, "%s\n", message); exit(2); }
}
int main(int argc, char** argv) {
  const char* pluginPath = nullptr;
  const char* outputPath = nullptr;
  for (int i = 1; i + 1 < argc; ++i) {
    if (!strcmp(argv[i], "--plugin")) pluginPath = argv[++i];
    else if (!strcmp(argv[i], "--output")) outputPath = argv[++i];
  }
  require(pluginPath && outputPath, "usage: profiler-host --plugin plugin.so --output events.jsonl");
  void* module = dlopen(pluginPath, RTLD_NOW | RTLD_LOCAL);
  const char* loadError = module ? nullptr : dlerror();
  require(module != nullptr, loadError ? loadError : "failed to load profiler plugin");
  auto getProfiler = reinterpret_cast<GetProfiler>(dlsym(module, "ncclProfiler_v7"));
  require(getProfiler != nullptr, "plugin does not export ncclProfiler_v7");
  ncclProfiler_v7_t* plugin = getProfiler();
  require(plugin && plugin->init && plugin->startEvent && plugin->stopEvent &&
    plugin->recordEventState && plugin->finalize, "plugin ABI function table is incomplete");
  void* context = nullptr;
  int mask = 0;
  require(plugin->init(&context, 1, &mask, "abi-harness", 1, 2, 0, logger) == ncclSuccess, "plugin init failed");
  require((mask & (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep)) ==
    (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep), "plugin activation mask incomplete");
  require(plugin->startEvent(context, nullptr, nullptr) == ncclInvalidArgument, "plugin accepted an invalid event");
  require(plugin->stopEvent(nullptr) == ncclInvalidArgument, "plugin accepted an invalid event handle");
  struct Scenario { uint64_t type; const char* name; } scenarios[] = {
    {ncclProfileColl, "collective"}, {ncclProfileProxyOp, "proxy_op"}, {ncclProfileProxyStep, "proxy_step"}
  };
  for (int i = 0; i < 3; ++i) {
    ncclProfilerEventDescr_v7_t description{};
    description.type = scenarios[i].type;
    description.rank = 0;
    if (i == 0) { description.coll.func = "AllReduce"; description.coll.count = 256; description.coll.datatype = "ncclInt"; }
    if (i == 1) { description.proxyOp.peer = 1; description.proxyOp.channelId = 2; description.proxyOp.nSteps = 4; description.proxyOp.chunkSize = 1024; description.proxyOp.isSend = 1; }
    if (i == 2) description.proxyStep.step = 3;
    void* handle = nullptr;
    require(plugin->startEvent(context, &handle, &description) == ncclSuccess && handle, "plugin startEvent failed");
    ncclProfilerEventStateArgs_v7_t state{};
    require(plugin->recordEventState(handle, ncclProfilerProxyStepSendGPUWait, &state) == ncclSuccess, "plugin recordEventState failed");
    require(plugin->stopEvent(handle) == ncclSuccess, "plugin stopEvent failed");
  }
  require(plugin->finalize(context) == ncclSuccess, "plugin finalize failed");
  dlclose(module);
  return 0;
}
