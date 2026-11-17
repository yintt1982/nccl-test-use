#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../nccl-tests-master/src/nccl_profiler.h"

typedef ncclProfiler_v7_t Plugin;

static void logger(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
static void require(bool condition, const char* message) {
  if (!condition) { fprintf(stderr, "%s\n", message); ExitProcess(2); }
}

int main(int argc, char** argv) {
  const char* pluginPath = nullptr;
  const char* outputPath = nullptr;
  for (int i = 1; i + 1 < argc; ++i) {
    if (!strcmp(argv[i], "--plugin")) pluginPath = argv[++i];
    else if (!strcmp(argv[i], "--output")) outputPath = argv[++i];
  }
  require(pluginPath && outputPath, "usage: profiler-host --plugin plugin.dll --output events.jsonl");
  HMODULE module = LoadLibraryA(pluginPath);
  require(module != nullptr, "failed to load profiler plugin");
  SetEnvironmentVariableA("NCCL_PROFILER_JSONL", outputPath);
  _putenv_s("NCCL_PROFILER_JSONL", outputPath);
  auto getProfiler = reinterpret_cast<Plugin* (*)()>(GetProcAddress(module, "ncclProfiler_v7"));
  require(getProfiler != nullptr, "plugin does not export ncclProfiler_v7");
  Plugin* plugin = getProfiler();
  require(plugin && plugin->init && plugin->startEvent && plugin->stopEvent &&
    plugin->recordEventState && plugin->finalize, "plugin ABI function table is incomplete");

  void* context = nullptr;
  int mask = 0;
  require(plugin->init(&context, 1, &mask, "abi-harness", 1, 2, 0, logger) == ncclSuccess,
    "plugin init failed");
  require((mask & (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep)) ==
    (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep), "plugin activation mask incomplete");
  void* invalidHandle = nullptr;
  require(plugin->startEvent(context, &invalidHandle, nullptr) == ncclInvalidArgument,
    "plugin accepted an invalid event descriptor");
  require(plugin->stopEvent(nullptr) == ncclInvalidArgument &&
    plugin->recordEventState(nullptr, ncclProfilerProxyStepSendGPUWait, nullptr) == ncclInvalidArgument,
    "plugin accepted an invalid callback handle");
  FILE* output = fopen(outputPath, "wb");
  require(output != nullptr, "cannot open JSONL output");

  struct Scenario { uint64_t type; int peer; uint8_t channel; int step; const char* name; } scenarios[] = {
    {ncclProfileColl, 1, 0, 0, "collective"},
    {ncclProfileProxyOp, 1, 2, 0, "proxy_op"},
    {ncclProfileProxyStep, 1, 2, 3, "proxy_step"}
  };
  for (int i = 0; i < 3; ++i) {
    ncclProfilerEventDescr_v7_t description = {};
    description.type = scenarios[i].type;
    description.rank = 0;
    description.parentObj = i ? reinterpret_cast<void*>(static_cast<uintptr_t>(i)) : nullptr;
    if (i == 0) {
      description.coll.seqNumber = 7;
      description.coll.func = "AllReduce";
      description.coll.count = 256;
      description.coll.datatype = "ncclInt";
      description.coll.algo = "Ring";
      description.coll.proto = "Simple";
    } else if (i == 1) {
      description.proxyOp.peer = scenarios[i].peer;
      description.proxyOp.channelId = scenarios[i].channel;
      description.proxyOp.nSteps = 4;
      description.proxyOp.chunkSize = 1024;
      description.proxyOp.isSend = 1;
    } else {
      description.proxyStep.step = scenarios[i].step;
    }
    void* handle = nullptr;
    require(plugin->startEvent(context, &handle, &description) == ncclSuccess && handle,
      "plugin startEvent failed");
    ncclProfilerEventStateArgs_v7_t state = {};
    require(plugin->recordEventState(handle, ncclProfilerProxyStepSendGPUWait, &state) == ncclSuccess,
      "plugin recordEventState failed");
    require(plugin->stopEvent(handle) == ncclSuccess, "plugin stopEvent failed");
  }
  fclose(output);
  require(plugin->finalize(context) == ncclSuccess, "plugin finalize failed");
  FreeLibrary(module);
  return 0;
}
