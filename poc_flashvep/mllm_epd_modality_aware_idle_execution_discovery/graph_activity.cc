// Collect graph envelopes only. Buffer callbacks do not parse or write records.
#include <cupti.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <set>
#include <utility>
#include <vector>

namespace {
constexpr size_t buffer_bytes = 1ULL << 20;

struct FinishedBuffer {
    CUcontext context;
    uint32_t stream;
    uint8_t* data;
    size_t valid;
};

std::vector<FinishedBuffer> buffers;
uint8_t* storage = nullptr;
std::atomic<size_t> requests{0}, completions{0};
std::atomic<uint64_t> overruns{0};
uint64_t records = 0, invalid = 0, dropped = 0, unexpected = 0;
bool active = false;

void CUPTIAPI provide(uint8_t** data, size_t* size, size_t* max_records) {
    const size_t slot = requests.fetch_add(1);
    *max_records = 0;
    if (slot >= buffers.size()) {
        ++overruns;
        *data = nullptr;
        *size = 0;
        return;
    }
    *data = storage + slot * buffer_bytes;
    *size = buffer_bytes;
}

void CUPTIAPI receive(CUcontext context, uint32_t stream, uint8_t* data, size_t size, size_t valid) {
    const size_t slot = completions.fetch_add(1);
    if (slot >= buffers.size() || valid > size || (valid && !data)) {
        ++overruns;
        return;
    }
    buffers[slot] = {context, stream, data, valid};
}
}

extern "C" int graph_activity_prepare(size_t bytes) {
    if (storage || bytes < 2 * buffer_bytes || bytes % buffer_bytes) return -1;
    try {
        buffers.resize(bytes / buffer_bytes);
    } catch (const std::bad_alloc&) {
        return -2;
    }
    void* address = nullptr;
    if (posix_memalign(&address, 64, bytes) != 0) return -3;
    storage = static_cast<uint8_t*>(address);
    std::memset(storage, 0, bytes);
    uint8_t already_zeroed = 1;
    size_t value_size = sizeof(already_zeroed);
    const auto status = cuptiActivitySetAttribute(
        CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER, &value_size, &already_zeroed);
    if (status != CUPTI_SUCCESS) return status;
    return cuptiActivityRegisterCallbacks(provide, receive);
}

extern "C" int graph_activity_enable() {
    if (!storage || active) return -4;
    const auto status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
    active = status == CUPTI_SUCCESS;
    return status;
}

extern "C" int graph_activity_timestamp(uint64_t* result) {
    return cuptiGetTimestamp(result);
}

extern "C" int graph_activity_close(const char* path) {
    if (!active) return -5;
    auto status = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
    if (status != CUPTI_SUCCESS) return status;
    active = false;
    status = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    if (status != CUPTI_SUCCESS) return status;

    FILE* output = std::fopen(path, "wx");
    if (!output) return -6;
    std::fprintf(output, "start_ns,end_ns,graph_id,correlation_id,start_device,start_context,launch_stream,end_device,end_context\n");
    int failure = 0;
    std::set<std::pair<CUcontext, uint32_t>> streams;
    const size_t received = completions.load();
    for (size_t i = 0; i < received && i < buffers.size(); ++i) {
        const auto& buffer = buffers[i];
        streams.emplace(buffer.context, buffer.stream);
        if (!buffer.valid) continue;
        CUpti_Activity* activity = nullptr;
        while (true) {
            status = cuptiActivityGetNextRecord(buffer.data, buffer.valid, &activity);
            if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
            if (status != CUPTI_SUCCESS) {
                failure = status;
                break;
            }
            if (activity->kind != CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
                ++unexpected;
                continue;
            }
            const auto* graph = reinterpret_cast<const CUpti_ActivityGraphTrace2*>(activity);
            if (!graph->start || graph->end <= graph->start) {
                ++invalid;
                continue;
            }
            std::fprintf(output, "%llu,%llu,%u,%u,%u,%u,%u,%u,%u\n",
                         static_cast<unsigned long long>(graph->start), static_cast<unsigned long long>(graph->end),
                         graph->graphId, graph->correlationId, graph->deviceId, graph->contextId,
                         graph->streamId, graph->endDeviceId, graph->endContextId);
            ++records;
        }
    }
    if (streams.empty()) streams.emplace(nullptr, 0);
    for (const auto& pair : streams) {
        size_t lost = 0;
        status = cuptiActivityGetNumDroppedRecords(pair.first, pair.second, &lost);
        if (status != CUPTI_SUCCESS) failure = status;
        dropped += lost;
    }
    const bool output_error = std::ferror(output) != 0;
    if (std::fclose(output) || output_error) return -7;
    if (overruns.load()) return -8;
    return failure;
}

extern "C" uint64_t graph_activity_records() { return records; }
extern "C" uint64_t graph_activity_invalid() { return invalid; }
extern "C" uint64_t graph_activity_dropped() { return dropped; }
extern "C" uint64_t graph_activity_unexpected() { return unexpected; }
extern "C" uint64_t graph_activity_overruns() { return overruns.load(); }
extern "C" uint64_t graph_activity_requested_buffers() { return requests.load(); }
extern "C" uint64_t graph_activity_completed_buffers() { return completions.load(); }
