// Fixed host buffers. No parsing, allocation, or file export in callbacks.
#include <cupti.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct Buffer {CUcontext context;uint32_t stream;uint8_t* data;size_t valid;};
static uint8_t* pool=nullptr;
static size_t block=8ULL*1024*1024,capacity=0;
static std::vector<Buffer> completed;
static std::atomic<size_t> requested{0},finished{0};
static std::atomic<int> failure{0};
static uint64_t dropped=0,invalid=0,all_kernels=0,exported=0;
static bool started=false,copies=false;

extern "C" int activity_options(int device_buffers,size_t bytes,size_t preallocate){
  uint8_t pinned=device_buffers?0:1;size_t size=sizeof(pinned);
  auto r=cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED,&size,&pinned);
  if(r!=CUPTI_SUCCESS)return r;
  size=sizeof(bytes);r=cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,&size,&bytes);
  if(r!=CUPTI_SUCCESS)return r;
  size=sizeof(preallocate);return cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE,&size,&preallocate);
}

static void CUPTIAPI request(uint8_t** data,size_t* bytes,size_t* maximum){
  *maximum=0;size_t i=requested.fetch_add(1);
  if(i>=capacity){*data=nullptr;*bytes=0;failure=-100;return;}
  *data=pool+i*block;*bytes=block;
}
static void CUPTIAPI complete(CUcontext c,uint32_t s,uint8_t* p,size_t,size_t n){
  size_t i=finished.fetch_add(1);
  if(i>=capacity){failure=-101;return;}
  completed[i]={c,s,p,n};
}
extern "C" int activity_begin(size_t pool_bytes,int include_copies){
  if(started||pool)return -102;
  capacity=pool_bytes/block;completed.resize(capacity);
  void* memory=nullptr;
  if(posix_memalign(&memory,64,capacity*block))return -103;
  pool=static_cast<uint8_t*>(memory);std::memset(pool,0,capacity*block);
  uint8_t zero=1;size_t size=sizeof(zero);
  auto r=cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER,&size,&zero);
  if(r!=CUPTI_SUCCESS)return r;
  r=cuptiActivityRegisterCallbacks(request,complete);if(r!=CUPTI_SUCCESS)return r;
  r=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);if(r!=CUPTI_SUCCESS)return r;
  copies=include_copies;
  if(copies){r=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);if(r!=CUPTI_SUCCESS)return r;}
  started=true;return 0;
}
extern "C" int activity_timestamp(uint64_t* value){return cuptiGetTimestamp(value);}
extern "C" int activity_end(const char* path,uint64_t left,uint64_t right){
  if(!started)return -104;
  auto r=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);if(r!=CUPTI_SUCCESS)return r;
  if(copies){r=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);if(r!=CUPTI_SUCCESS)return r;}
  r=cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);if(r!=CUPTI_SUCCESS)return r;
  FILE* f=std::fopen(path,"w");if(!f)return -105;
  std::fprintf(f,"kind,start_ns,end_ns,graph_node,graph_id,device,context,stream,correlation,bytes,copy_kind,name_id\n");
  std::unordered_map<std::string,unsigned> names;std::vector<std::string> ordered;
  std::set<std::pair<CUcontext,uint32_t>> streams;
  size_t total=finished.load();if(total>capacity)total=capacity;
  for(size_t i=0;i<total;++i){
    const auto& b=completed[i];streams.insert({b.context,b.stream});CUpti_Activity* a=nullptr;
    while(true){
      r=cuptiActivityGetNextRecord(b.data,b.valid,&a);
      if(r==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
      if(r!=CUPTI_SUCCESS){failure=r;break;}
      if(a->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
        auto* k=reinterpret_cast<CUpti_ActivityKernel9*>(a);++all_kernels;
        if(!k->start||k->end<=k->start){++invalid;continue;}
        if(k->end<=left||k->start>=right)continue;
        std::string name=k->name?k->name:"UNKNOWN";
        auto it=names.find(name);unsigned id;
        if(it==names.end()){id=ordered.size();names[name]=id;ordered.push_back(name);}else id=it->second;
        std::fprintf(f,"kernel,%llu,%llu,%llu,%u,%u,%u,%u,%u,0,0,%u\n",
          (unsigned long long)k->start,(unsigned long long)k->end,(unsigned long long)k->graphNodeId,
          k->graphId,k->deviceId,k->contextId,k->streamId,k->correlationId,id);++exported;
      }else if(a->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
        auto* m=reinterpret_cast<CUpti_ActivityMemcpy6*>(a);
        if(!m->start||m->end<=m->start){++invalid;continue;}
        if(m->end<=left||m->start>=right)continue;
        std::fprintf(f,"memcpy,%llu,%llu,%llu,%u,%u,%u,%u,%u,%llu,%u,0\n",
          (unsigned long long)m->start,(unsigned long long)m->end,(unsigned long long)m->graphNodeId,
          m->graphId,m->deviceId,m->contextId,m->streamId,m->correlationId,(unsigned long long)m->bytes,m->copyKind);
      }
    }
  }
  for(const auto& s:streams){size_t lost=0;r=cuptiActivityGetNumDroppedRecords(s.first,s.second,&lost);if(r!=CUPTI_SUCCESS)failure=r;dropped+=lost;}
  std::fclose(f);
  std::string name_path=std::string(path)+".names.csv";f=std::fopen(name_path.c_str(),"w");
  if(!f)return -106;
  std::fprintf(f,"name_id,kernel_name\n");
  for(unsigned i=0;i<ordered.size();++i){
    std::fprintf(f,"%u,\"",i);for(char c:ordered[i]){if(c=='\"')std::fputc('\"',f);std::fputc(c,f);}std::fprintf(f,"\"\n");
  }
  std::fclose(f);started=false;return failure.load();
}
extern "C" uint64_t activity_dropped(){return dropped;}
extern "C" uint64_t activity_invalid(){return invalid;}
extern "C" uint64_t activity_count(){return exported;}
extern "C" uint64_t activity_all_kernels(){return all_kernels;}
extern "C" uint64_t activity_requested_buffers(){return requested.load();}
extern "C" uint64_t activity_completed_buffers(){return finished.load();}
extern "C" uint64_t activity_reserved_host_bytes(){return capacity*block;}
