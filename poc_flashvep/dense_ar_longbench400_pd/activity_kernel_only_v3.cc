// Record GPU kernel and copy activity. Do not trace driver/runtime APIs.
#include <cupti.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct SavedBuffer {CUcontext context;uint32_t stream;uint8_t* data;size_t valid;};
static std::vector<SavedBuffer> buffers;
static uint8_t* pool=nullptr;
static size_t block=8ULL*1024*1024,capacity=0;
static std::atomic<size_t> requested{0},completed{0};
static std::atomic<int> failure{0};
static uint64_t dropped=0,invalid=0,kernels=0;
static bool started=false,copies=false;
static void CUPTIAPI request(uint8_t** data,size_t* bytes,size_t* maximum){
  *maximum=0;size_t i=requested.fetch_add(1);
  if(i>=capacity){*data=nullptr;*bytes=0;failure=-100;return;}
  *data=pool+i*block;*bytes=block;
}
static void CUPTIAPI complete(CUcontext c,uint32_t s,uint8_t* p,size_t,size_t n){
  size_t i=completed.fetch_add(1);
  if(i>=capacity){failure=-101;return;}
  buffers[i]={c,s,p,n};
}
extern "C" int activity_graph_mode(int value){return value?-109:0;}
extern "C" int activity_prepare(size_t bytes){
  if(pool)return -102;
  capacity=bytes/block;buffers.resize(capacity);void* p=nullptr;
  if(posix_memalign(&p,64,capacity*block))return -103;
  pool=static_cast<uint8_t*>(p);std::memset(pool,0,capacity*block);
  uint8_t zero=1;size_t size=sizeof(zero);
  auto r=cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER,&size,&zero);
  return r==CUPTI_SUCCESS?cuptiActivityRegisterCallbacks(request,complete):r;
}
extern "C" int activity_enable(int include_copies){
  if(!pool||started)return -108;
  auto r=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);if(r!=CUPTI_SUCCESS)return r;
  copies=include_copies;
  if(copies)for(auto k:{CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_MEMCPY2,CUPTI_ACTIVITY_KIND_MEMSET}){
    r=cuptiActivityEnable(k);if(r!=CUPTI_SUCCESS)return r;
  }
  started=true;return 0;
}
extern "C" int activity_disable(){
  if(!started)return -104;
  auto r=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);if(r!=CUPTI_SUCCESS)return r;
  if(copies)for(auto k:{CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_MEMCPY2,CUPTI_ACTIVITY_KIND_MEMSET}){
    r=cuptiActivityDisable(k);if(r!=CUPTI_SUCCESS)return r;
  }
  started=false;return 0;
}
extern "C" int activity_push_id(uint64_t){return -120;}
extern "C" int activity_pop_id(uint64_t*){return -120;}
extern "C" int activity_timestamp(uint64_t* value){return cuptiGetTimestamp(value);}
extern "C" int activity_end(const char* path,uint64_t left,uint64_t right){
  if(started)return -121;
  auto r=cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);if(r!=CUPTI_SUCCESS)return r;
  FILE* out=std::fopen(path,"w");if(!out)return -105;
  std::fprintf(out,"kind,start_ns,end_ns,graph_node,graph_id,device,context,stream,correlation,bytes,copy_kind,name_id,src_device,dst_device,api_id\n");
  std::unordered_map<std::string,unsigned> names;std::vector<std::string> ordered;
  std::set<std::pair<CUcontext,uint32_t>> streams;
  size_t n=completed.load();if(n>capacity)n=capacity;
  for(size_t i=0;i<n;++i){
    const auto& b=buffers[i];streams.insert({b.context,b.stream});CUpti_Activity* record=nullptr;
    while(true){
      r=cuptiActivityGetNextRecord(b.data,b.valid,&record);
      if(r==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
      if(r!=CUPTI_SUCCESS){failure=r;break;}
      uint64_t start=0,end=0,bytes=0,node=0;uint32_t device=0,context=0,stream=0,corr=0,graph=0,copy=0,id=0,src=0,dst=0;
      const char* kind=nullptr;
      if(record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
        auto* k=reinterpret_cast<CUpti_ActivityKernel9*>(record);kind="kernel";
        start=k->start;end=k->end;device=k->deviceId;context=k->contextId;stream=k->streamId;corr=k->correlationId;node=k->graphNodeId;graph=k->graphId;
        std::string name=k->name?k->name:"UNKNOWN";auto found=names.find(name);
        if(found==names.end()){id=ordered.size();names[name]=id;ordered.push_back(name);}else id=found->second;
        ++kernels;
      }else if(record->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
        auto* m=reinterpret_cast<CUpti_ActivityMemcpy6*>(record);kind="memcpy";
        start=m->start;end=m->end;device=m->deviceId;context=m->contextId;stream=m->streamId;corr=m->correlationId;node=m->graphNodeId;graph=m->graphId;bytes=m->bytes;copy=m->copyKind;src=dst=device;
      }else if(record->kind==CUPTI_ACTIVITY_KIND_MEMCPY2){
        auto* m=reinterpret_cast<CUpti_ActivityMemcpyPtoP4*>(record);kind="memcpy2";
        start=m->start;end=m->end;device=m->deviceId;context=m->contextId;stream=m->streamId;corr=m->correlationId;node=m->graphNodeId;graph=m->graphId;bytes=m->bytes;copy=m->copyKind;src=m->srcDeviceId;dst=m->dstDeviceId;
      }else if(record->kind==CUPTI_ACTIVITY_KIND_MEMSET){
        auto* m=reinterpret_cast<CUpti_ActivityMemset4*>(record);kind="memset";
        start=m->start;end=m->end;device=m->deviceId;context=m->contextId;stream=m->streamId;corr=m->correlationId;node=m->graphNodeId;graph=m->graphId;bytes=m->bytes;
      }else {++invalid;continue;}
      if(!start||end<=start){++invalid;continue;}
      if(end<=left||start>=right)continue;
      std::fprintf(out,"%s,%llu,%llu,%llu,%u,%u,%u,%u,%u,%llu,%u,%u,%u,%u,\n",kind,
        (unsigned long long)start,(unsigned long long)end,(unsigned long long)node,graph,device,context,stream,corr,(unsigned long long)bytes,copy,id,src,dst);
    }
  }
  std::fclose(out);
  for(const auto& s:streams){size_t lost=0;r=cuptiActivityGetNumDroppedRecords(s.first,s.second,&lost);if(r!=CUPTI_SUCCESS)failure=r;dropped+=lost;}
  FILE* f=std::fopen((std::string(path)+".names.csv").c_str(),"w");if(!f)return -106;
  std::fprintf(f,"name_id,kernel_name\n");
  for(unsigned i=0;i<ordered.size();++i){std::fprintf(f,"%u,\"",i);for(char c:ordered[i]){if(c=='\"')std::fputc('\"',f);std::fputc(c,f);}std::fprintf(f,"\"\n");}
  std::fclose(f);return failure.load();
}
extern "C" uint64_t activity_dropped(){return dropped;}
extern "C" uint64_t activity_invalid(){return invalid;}
extern "C" uint64_t activity_count(){return kernels;}
extern "C" uint64_t activity_all_kernels(){return kernels;}
extern "C" uint64_t activity_requested_buffers(){return requested.load();}
extern "C" uint64_t activity_completed_buffers(){return completed.load();}
extern "C" uint64_t activity_reserved_host_bytes(){return capacity*block;}
