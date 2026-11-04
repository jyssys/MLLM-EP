// Activity-only diagnostic. No CUDA API callback subscription or model changes.
#include <cupti.h>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <vector>

struct Record {
  uint64_t start, end, node;
  uint32_t device, context, stream;
};
static std::mutex lock;
static std::vector<Record> records;
static std::atomic<uint64_t> dropped{0}, invalid{0};
static std::atomic<int> failure{0};
static bool started = false;

static void CUPTIAPI request(uint8_t **buffer, size_t *size, size_t *maximum) {
  *size = 16 * 1024 * 1024;
  *maximum = 0;
  void *memory = nullptr;
  if (posix_memalign(&memory, 8, *size)) {
    failure = -1;
    *size = 0;
  }
  *buffer = static_cast<uint8_t *>(memory);
}

static void CUPTIAPI complete(CUcontext context, uint32_t stream,
                              uint8_t *buffer, size_t, size_t valid) {
  CUpti_Activity *activity = nullptr;
  std::vector<Record> local;
  for (;;) {
    CUptiResult result = cuptiActivityGetNextRecord(buffer, valid, &activity);
    if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
    if (result != CUPTI_SUCCESS) { failure = result; break; }
    if (activity->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
    auto *kernel = reinterpret_cast<CUpti_ActivityKernel9 *>(activity);
    if (!kernel->start || kernel->end <= kernel->start) { ++invalid; continue; }
    local.push_back({kernel->start, kernel->end, kernel->graphNodeId,
                     kernel->deviceId, kernel->contextId, kernel->streamId});
  }
  {
    std::lock_guard<std::mutex> guard(lock);
    records.insert(records.end(), local.begin(), local.end());
  }
  size_t lost = 0;
  CUptiResult result = cuptiActivityGetNumDroppedRecords(context, stream, &lost);
  if (result != CUPTI_SUCCESS) failure = result;
  dropped += lost;
  std::free(buffer);
}

extern "C" int activity_begin() {
  if (started) return -2;
  records.reserve(500000);
  CUptiResult result = cuptiActivityRegisterCallbacks(request, complete);
  if (result != CUPTI_SUCCESS) return result;
  result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result == CUPTI_SUCCESS) started = true;
  return result;
}

extern "C" int activity_timestamp(uint64_t *timestamp) {
  return cuptiGetTimestamp(timestamp);
}

extern "C" int activity_end(const char *path) {
  if (!started) return -3;
  CUptiResult result = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result != CUPTI_SUCCESS) return result;
  result = cuptiActivityFlushAll(0);
  if (result != CUPTI_SUCCESS) return result;
  std::lock_guard<std::mutex> guard(lock);
  std::ofstream output(path);
  if (!output) return -4;
  output << "start_ns,end_ns,graph_node,device,context,stream\n";
  for (const auto &row : records)
    output << row.start << ',' << row.end << ',' << row.node << ','
           << row.device << ',' << row.context << ',' << row.stream << '\n';
  started = false;
  return failure.load();
}

extern "C" uint64_t activity_dropped() { return dropped.load(); }
extern "C" uint64_t activity_invalid() { return invalid.load(); }
extern "C" size_t activity_count() { return records.size(); }
