// Activity records stay in memory. Serving never exports or flushes them.
#include <cupti.h>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <vector>
struct Record { uint64_t start,end,node; uint32_t kind,device,context,stream; };
static std::mutex guard;
static std::vector<Record> records;
static std::vector<Record> bad_records;
static std::atomic<uint64_t> dropped{0},invalid{0};
static std::atomic<int> failure{0};
static std::atomic<uint64_t> requests{0},completions{0},valid_bytes{0},parsed{0};
static bool registered=false;
static int active=0;
static void CUPTIAPI request(uint8_t **buffer,size_t *size,size_t *maximum) {
  ++requests;
  *size=8*1024*1024;*maximum=0;void *memory=nullptr;
  if(posix_memalign(&memory,8,*size)){failure=-1;*size=0;}
  *buffer=static_cast<uint8_t *>(memory);
}
static void CUPTIAPI complete(CUcontext context,uint32_t stream,uint8_t *buffer,size_t,size_t valid) {
  ++completions;valid_bytes+=valid;
  CUpti_Activity *item=nullptr;std::vector<Record> local,bad;
  for(;;){
    auto result=cuptiActivityGetNextRecord(buffer,valid,&item);
    if(result==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
    if(result!=CUPTI_SUCCESS){failure=result;break;}
    ++parsed;
    Record value{};value.kind=item->kind;
    if(item->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
      auto *kernel=reinterpret_cast<CUpti_ActivityKernel9 *>(item);
      value.start=kernel->start;value.end=kernel->end;value.node=kernel->graphNodeId;
      value.device=kernel->deviceId;value.context=kernel->contextId;value.stream=kernel->streamId;
    } else if(item->kind==CUPTI_ACTIVITY_KIND_GRAPH_TRACE){
      auto *graph=reinterpret_cast<CUpti_ActivityGraphTrace2 *>(item);
      value.start=graph->start;value.end=graph->end;value.node=graph->graphId;
      value.device=graph->deviceId;value.context=graph->contextId;value.stream=graph->streamId;
    } else continue;
    if(!value.start||value.end<=value.start){++invalid;bad.push_back(value);continue;}
    local.push_back(value);
  }
  {std::lock_guard<std::mutex> lock(guard);records.insert(records.end(),local.begin(),local.end());
    bad_records.insert(bad_records.end(),bad.begin(),bad.end());}
  size_t lost=0;auto status=cuptiActivityGetNumDroppedRecords(context,stream,&lost);
  if(status!=CUPTI_SUCCESS)failure=status;dropped+=lost;std::free(buffer);
}
extern "C" int activity_begin(int mode){
  if(active)return -2;
  if(!registered){
    auto status=cuptiActivityRegisterCallbacks(request,complete);if(status!=CUPTI_SUCCESS)return status;
    registered=true;records.reserve(200000);
  }
  if(mode&2){auto s=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);if(s!=CUPTI_SUCCESS)return s;active|=2;}
  if(mode&1){auto s=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);if(s!=CUPTI_SUCCESS)return s;active|=1;}
  return 0;
}
extern "C" int activity_pause(){
  if(active&1){auto s=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);if(s!=CUPTI_SUCCESS)return s;active&=~1;}
  if(active&2){auto s=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);if(s!=CUPTI_SUCCESS)return s;active&=~2;}
  return 0;
}
extern "C" int activity_reset(){
  if(active)return -2;auto s=cuptiActivityFlushAll(0);if(s!=CUPTI_SUCCESS)return s;
  std::lock_guard<std::mutex> lock(guard);records.clear();bad_records.clear();dropped=0;invalid=0;failure=0;return 0;
}
extern "C" int activity_timestamp(uint64_t *stamp){return cuptiGetTimestamp(stamp);}
static int end_to_file(const char *path,uint32_t flags,bool flush_first=false){
  int s=0;if(flush_first){s=cuptiActivityFlushAll(flags);if(s)return s;}
  s=activity_pause();if(s)return s;s=cuptiActivityFlushAll(flags);if(s)return s;
  std::lock_guard<std::mutex> lock(guard);std::ofstream output(path);if(!output)return -4;
  output<<"start_ns,end_ns,kind,node,device,context,stream\n";
  for(auto r:records)output<<r.start<<','<<r.end<<','<<r.kind<<','<<r.node<<','<<r.device<<','<<r.context<<','<<r.stream<<'\n';
  std::ofstream invalid_output(std::string(path)+".invalid.csv");
  invalid_output<<"start_ns,end_ns,kind,node,device,context,stream\n";
  for(auto r:bad_records)invalid_output<<r.start<<','<<r.end<<','<<r.kind<<','<<r.node<<','<<r.device<<','<<r.context<<','<<r.stream<<'\n';
  return failure.load();
}
extern "C" int activity_end(const char *path){return end_to_file(path,0);}
extern "C" int activity_end_forced(const char *path){return end_to_file(path,CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);}
extern "C" int activity_end_drain_first(const char *path){return end_to_file(path,CUPTI_ACTIVITY_FLAG_FLUSH_FORCED,true);}
extern "C" uint64_t activity_dropped(){return dropped.load();}
extern "C" uint64_t activity_invalid(){return invalid.load();}
extern "C" uint64_t activity_requests(){return requests.load();}
extern "C" uint64_t activity_completions(){return completions.load();}
extern "C" uint64_t activity_bytes(){return valid_bytes.load();}
extern "C" uint64_t activity_parsed(){return parsed.load();}
extern "C" int activity_version(){uint32_t value=0;return cuptiGetVersion(&value)==CUPTI_SUCCESS?value:-1;}
