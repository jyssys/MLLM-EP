// Retain activity buffers during serving. Parse them only after serving ends.
#include <cupti.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

struct Buffer { CUcontext context; uint32_t stream; uint8_t *data; size_t valid; };
static std::mutex mutex;
static std::vector<Buffer> buffers;
static std::atomic<uint64_t> allocated{0}, dropped{0}, invalid{0}, count{0};
static std::atomic<int> failure{0};
static bool started = false;
static constexpr size_t size_bytes = 16 * 1024 * 1024;
static constexpr uint64_t maximum_bytes = 8ULL * 1024 * 1024 * 1024;

static void CUPTIAPI request(uint8_t **buffer, size_t *size, size_t *maximum) {
  *maximum = 0; *size = size_bytes; *buffer = nullptr;
  if (allocated.fetch_add(size_bytes) + size_bytes > maximum_bytes) {
    failure = -10; *size = 0; return;
  }
  void *memory = nullptr;
  if (posix_memalign(&memory, 8, size_bytes)) { failure = -11; *size = 0; return; }
  *buffer = static_cast<uint8_t *>(memory);
}

static void CUPTIAPI complete(CUcontext context, uint32_t stream,
                              uint8_t *buffer, size_t, size_t valid) {
  std::lock_guard<std::mutex> guard(mutex);
  buffers.push_back({context, stream, buffer, valid});
}

extern "C" int activity_begin() {
  if (started) return -2;
  buffers.reserve(1024);
  auto result = cuptiActivityRegisterCallbacks(request, complete);
  if (result != CUPTI_SUCCESS) return result;
  result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result != CUPTI_SUCCESS) return result;
  // Kernel-only revision. Memcpy activity remains a separate unmeasured field.
  if (result == CUPTI_SUCCESS) started = true;
  return result;
}
extern "C" int activity_timestamp(uint64_t *timestamp) { return cuptiGetTimestamp(timestamp); }
extern "C" int activity_end(const char *path) {
  if (!started) return -3;
  auto result = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result != CUPTI_SUCCESS) return result;
  result = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  if (result != CUPTI_SUCCESS) return result;
  std::lock_guard<std::mutex> guard(mutex);
  FILE *out = std::fopen(path, "w");
  if (!out) return -4;
  std::set<std::pair<CUcontext, uint32_t>> streams;
  std::fprintf(out, "kind,start_ns,end_ns,graph_node,graph_id,device,context,stream,correlation,bytes,copy_kind,communication\n");
  for (const auto &buffer : buffers) {
    streams.insert({buffer.context, buffer.stream});
    CUpti_Activity *activity = nullptr;
    for (;;) {
      result = cuptiActivityGetNextRecord(buffer.data, buffer.valid, &activity);
      if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
      if (result != CUPTI_SUCCESS) { failure = result; break; }
      if (activity->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        auto *k = reinterpret_cast<CUpti_ActivityKernel9 *>(activity);
        if (!k->start || k->end <= k->start) { ++invalid; continue; }
        bool comm = k->name && (std::strstr(k->name,"nccl") || std::strstr(k->name,"nixl"));
        std::fprintf(out,"kernel,%llu,%llu,%llu,%u,%u,%u,%u,%u,0,0,%d\n",
          (unsigned long long)k->start,(unsigned long long)k->end,
          (unsigned long long)k->graphNodeId,k->graphId,k->deviceId,
          k->contextId,k->streamId,k->correlationId,comm ? 1 : 0);
        ++count;
      } else if (activity->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
        auto *m = reinterpret_cast<CUpti_ActivityMemcpy6 *>(activity);
        if (!m->start || m->end <= m->start) { ++invalid; continue; }
        std::fprintf(out,"memcpy,%llu,%llu,%llu,%u,%u,%u,%u,%u,%llu,%u,0\n",
          (unsigned long long)m->start,(unsigned long long)m->end,
          (unsigned long long)m->graphNodeId,m->graphId,m->deviceId,
          m->contextId,m->streamId,m->correlationId,(unsigned long long)m->bytes,m->copyKind);
      }
    }
    std::free(buffer.data);
  }
  for (const auto &stream : streams) {
    size_t lost = 0;
    result = cuptiActivityGetNumDroppedRecords(stream.first, stream.second, &lost);
    if (result != CUPTI_SUCCESS) failure = result;
    dropped += lost;
  }
  std::fclose(out); buffers.clear(); started = false;
  return failure.load();
}
extern "C" uint64_t activity_dropped() { return dropped.load(); }
extern "C" uint64_t activity_invalid() { return invalid.load(); }
extern "C" uint64_t activity_count() { return count.load(); }
extern "C" uint64_t activity_allocated() { return allocated.load(); }
