// Reuse the activity-only CUPTI design. Flush only after serving has drained.
#include <cupti.h>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <vector>
struct Record { uint64_t start,end,node; uint32_t device,context,stream; };
static std::mutex lock;
static std::vector<Record> records;
static std::atomic<uint64_t> dropped{0},invalid{0};
static std::atomic<int> failure{0};
static bool registered=false,active=false;
static void CUPTIAPI request(uint8_t **buffer,size_t *size,size_t *maximum) {
  *size=16*1024*1024;*maximum=0;void *memory=nullptr;
  if(posix_memalign(&memory,8,*size)){failure=-1;*size=0;}
  *buffer=static_cast<uint8_t *>(memory);
}
static void CUPTIAPI complete(CUcontext context,uint32_t stream,uint8_t *buffer,size_t,size_t valid) {
  CUpti_Activity *activity=nullptr;std::vector<Record> local;
  for(;;){
    auto result=cuptiActivityGetNextRecord(buffer,valid,&activity);
    if(result==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
    if(result!=CUPTI_SUCCESS){failure=result;break;}
    if(activity->kind!=CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)continue;
    auto *kernel=reinterpret_cast<CUpti_ActivityKernel9 *>(activity);
    if(!kernel->start||kernel->end<=kernel->start){++invalid;continue;}
    local.push_back({kernel->start,kernel->end,kernel->graphNodeId,kernel->deviceId,kernel->contextId,kernel->streamId});
  }
  {std::lock_guard<std::mutex> guard(lock);records.insert(records.end(),local.begin(),local.end());}
  size_t lost=0;auto result=cuptiActivityGetNumDroppedRecords(context,stream,&lost);
  if(result!=CUPTI_SUCCESS)failure=result;dropped+=lost;std::free(buffer);
}
extern "C" int activity_begin(){
  if(active)return -2;
  if(!registered){
    auto status=cuptiActivityRegisterCallbacks(request,complete);
    if(status!=CUPTI_SUCCESS)return status;
    registered=true;records.reserve(500000);
  }
  auto status=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if(status==CUPTI_SUCCESS)active=true;return status;
}
extern "C" int activity_pause(){
  if(!active)return 0;
  auto status=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if(status==CUPTI_SUCCESS)active=false;return status;
}
extern "C" int activity_reset(){
  if(active)return -2;
  auto status=cuptiActivityFlushAll(0);if(status!=CUPTI_SUCCESS)return status;
  std::lock_guard<std::mutex> guard(lock);records.clear();dropped=0;invalid=0;failure=0;return 0;
}
extern "C" int activity_timestamp(uint64_t *timestamp){return cuptiGetTimestamp(timestamp);}
extern "C" int activity_end(const char *path){
  int status=activity_pause();if(status)return status;
  status=cuptiActivityFlushAll(0);if(status)return status;
  std::lock_guard<std::mutex> guard(lock);std::ofstream output(path);if(!output)return -4;
  output<<"start_ns,end_ns,graph_node,device,context,stream\n";
  for(const auto &r:records)output<<r.start<<','<<r.end<<','<<r.node<<','<<r.device<<','<<r.context<<','<<r.stream<<'\n';
  return failure.load();
}
extern "C" uint64_t activity_dropped(){return dropped.load();}
extern "C" uint64_t activity_invalid(){return invalid.load();}
extern "C" size_t activity_count(){return records.size();}
