// Concurrent kernel activity only. Callback work is restricted to fixed buffers.
#include <cupti.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr size_t block_bytes = 4ULL << 20;
struct Complete {
    CUcontext context;
    uint32_t stream;
    uint8_t *data;
    size_t length;
};
std::vector<Complete> buffers;
std::atomic<size_t> requested{0}, completed{0};
uint8_t *storage = nullptr;
uint64_t dropped = 0, invalid = 0, exported = 0;
bool started = false;

void CUPTIAPI request(uint8_t **ptr, size_t *bytes, size_t *max_records) {
    const size_t i = requested.fetch_add(1);
    *max_records = 0;
    *ptr = i >= buffers.size() ? nullptr : storage + i * block_bytes;
    *bytes = *ptr == nullptr ? 0 : block_bytes;
}

void CUPTIAPI complete(CUcontext context, uint32_t stream, uint8_t *data, size_t, size_t valid) {
    const size_t i = completed.fetch_add(1);
    if (i < buffers.size()) {
        buffers[i] = {context, stream, data, valid};
    }
}
}

extern "C" int pd_activity_start(size_t size_mib) {
    if (started || storage != nullptr || size_mib < 8) return -1;
    const size_t capacity = (size_mib << 20) / block_bytes;
    void *host = nullptr;
    if (posix_memalign(&host, 64, capacity * block_bytes)) return -2;
    storage = static_cast<uint8_t *>(host);
    std::memset(storage, 0, capacity * block_bytes);
    buffers.resize(capacity);
    auto rc = cuptiActivityRegisterCallbacks(request, complete);
    if (rc != CUPTI_SUCCESS) return rc;
    rc = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    started = rc == CUPTI_SUCCESS;
    return rc;
}

extern "C" int pd_activity_timestamp(uint64_t *out) {
    return cuptiGetTimestamp(out);
}

extern "C" int pd_activity_stop(const char *filename) {
    if (!started) return -3;
    auto rc = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (rc != CUPTI_SUCCESS) return rc;
    rc = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    if (rc != CUPTI_SUCCESS) return rc;
    started = false;
    auto *file = std::fopen(filename, "w");
    if (file == nullptr) return -4;
    std::fprintf(file, "start_ns,end_ns,device,stream,grid_x,grid_y,grid_z,block_x,block_y,block_z,name\n");
    const auto count = completed.load();
    for (size_t i = 0; i < count && i < buffers.size(); ++i) {
        const auto &b = buffers[i];
        size_t missing = 0;
        rc = cuptiActivityGetNumDroppedRecords(b.context, b.stream, &missing);
        if (rc == CUPTI_SUCCESS) dropped += missing;
        CUpti_Activity *rec = nullptr;
        while (cuptiActivityGetNextRecord(b.data, b.length, &rec) == CUPTI_SUCCESS) {
            if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
            auto *k = reinterpret_cast<CUpti_ActivityKernel9 *>(rec);
            if (!k->start || k->end <= k->start || !k->name) {
                ++invalid;
                continue;
            }
            std::fprintf(file, "%llu,%llu,%u,%u,%d,%d,%d,%d,%d,%d,",
                         static_cast<unsigned long long>(k->start), static_cast<unsigned long long>(k->end),
                         k->deviceId, k->streamId, k->gridX, k->gridY, k->gridZ, k->blockX, k->blockY, k->blockZ);
            std::fputc('"', file);
            for (const char *c = k->name; *c; ++c) {
                if (*c == '"') std::fputc('"', file);
                std::fputc(*c, file);
            }
            std::fputs("\"\n", file);
            ++exported;
        }
    }
    std::fclose(file);
    if (count > buffers.size() || requested.load() > buffers.size()) return -5;
    return 0;
}

extern "C" uint64_t pd_activity_exported() { return exported; }
extern "C" uint64_t pd_activity_invalid() { return invalid; }
extern "C" uint64_t pd_activity_dropped() { return dropped; }
