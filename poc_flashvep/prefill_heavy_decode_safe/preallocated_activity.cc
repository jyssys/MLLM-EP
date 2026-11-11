// Fixed host buffers. No parsing, allocation, or file export in callbacks.
#include <cupti.h>
#include <cupti_driver_cbid.h>
#include <cupti_runtime_cbid.h>
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
static bool started=false,copies=false,graph_mode=false;
static uint64_t all_graphs=0;
static CUpti_ActivityKind kernel_kind=CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
static FILE* invalid_file=nullptr;
static void invalid_record(const char* kind,uint64_t start,uint64_t end,uint32_t device,
                           uint32_t context,uint32_t stream,uint64_t bytes,
                           uint32_t end_device,uint32_t end_context,
                           uint32_t graph_id=0,uint32_t correlation=0){
  ++invalid;
  std::fprintf(invalid_file,"%s,%llu,%llu,%u,%u,%u,%llu,%u,%u,%u,%u\n",kind,
    (unsigned long long)start,(unsigned long long)end,device,context,stream,
    (unsigned long long)bytes,end_device,end_context,graph_id,correlation);
}

extern "C" int activity_kernel_mode(int serial){
  if(started||pool)return -107;
  kernel_kind=serial?CUPTI_ACTIVITY_KIND_KERNEL:CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
  return 0;
}
extern "C" int activity_graph_mode(int enabled){
  if(started||pool)return -109;
  graph_mode=enabled;return 0;
}

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
extern "C" int activity_prepare(size_t pool_bytes){
  if(started||pool)return -102;
  capacity=pool_bytes/block;completed.resize(capacity);
  void* memory=nullptr;
  if(posix_memalign(&memory,64,capacity*block))return -103;
  pool=static_cast<uint8_t*>(memory);std::memset(pool,0,capacity*block);
  uint8_t zero=1;size_t size=sizeof(zero);
  auto r=cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER,&size,&zero);
  if(r!=CUPTI_SUCCESS)return r;
  r=cuptiActivityRegisterCallbacks(request,complete);if(r!=CUPTI_SUCCESS)return r;
  return 0;
}
extern "C" int activity_enable(int include_copies){
  if(!pool||started)return -108;
  auto r=CUPTI_SUCCESS;
  if(graph_mode){
    r=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);if(r!=CUPTI_SUCCESS)return r;
    r=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);if(r!=CUPTI_SUCCESS)return r;
    r=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);if(r!=CUPTI_SUCCESS)return r;
  }
  r=cuptiActivityEnable(kernel_kind);if(r!=CUPTI_SUCCESS)return r;
  copies=include_copies;
  if(copies){
    for(auto kind:{CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_MEMCPY2,CUPTI_ACTIVITY_KIND_MEMSET}){
      r=cuptiActivityEnable(kind);if(r!=CUPTI_SUCCESS)return r;
    }
  }
  started=true;return 0;
}
extern "C" int activity_begin(size_t pool_bytes,int include_copies){
  auto r=activity_prepare(pool_bytes);return r?r:activity_enable(include_copies);
}
extern "C" int activity_disable(){
  if(!started)return -104;
  auto r=cuptiActivityDisable(kernel_kind);if(r!=CUPTI_SUCCESS)return r;
  if(graph_mode){
    r=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);if(r!=CUPTI_SUCCESS)return r;
    r=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);if(r!=CUPTI_SUCCESS)return r;
    r=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);if(r!=CUPTI_SUCCESS)return r;
  }
  if(copies){
    for(auto kind:{CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_MEMCPY2,CUPTI_ACTIVITY_KIND_MEMSET}){
      r=cuptiActivityDisable(kind);if(r!=CUPTI_SUCCESS)return r;
    }
  }
  started=false;return 0;
}
extern "C" int activity_timestamp(uint64_t* value){return cuptiGetTimestamp(value);}
extern "C" int activity_end(const char* path,uint64_t left,uint64_t right){
  if(!pool)return -104;
  auto r=CUPTI_SUCCESS;
  if(started){auto code=activity_disable();if(code)return code;}
  r=cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);if(r!=CUPTI_SUCCESS)return r;
  FILE* f=std::fopen(path,"w");if(!f)return -105;
  std::string invalid_path=std::string(path)+".invalid.csv";
  invalid_file=std::fopen(invalid_path.c_str(),"w");if(!invalid_file){std::fclose(f);return -110;}
  std::fprintf(invalid_file,"kind,start_ns,end_ns,device,context,stream,bytes,end_device,end_context,graph_id,correlation\n");
  std::fprintf(f,"kind,start_ns,end_ns,graph_node,graph_id,device,context,stream,correlation,bytes,copy_kind,name_id,src_device,dst_device,api_id\n");
  std::unordered_map<std::string,unsigned> names;std::vector<std::string> ordered;
  std::set<std::pair<CUcontext,uint32_t>> streams;
  std::unordered_map<unsigned,uint64_t> kind_counts;
  std::set<uint32_t> graph_correlations;
  size_t total=finished.load();if(total>capacity)total=capacity;
  for(size_t i=0;i<total;++i){
    const auto& b=completed[i];CUpti_Activity* a=nullptr;
    while(cuptiActivityGetNextRecord(b.data,b.valid,&a)==CUPTI_SUCCESS){
      if(a->kind==CUPTI_ACTIVITY_KIND_GRAPH_TRACE)
        graph_correlations.insert(reinterpret_cast<CUpti_ActivityGraphTrace2*>(a)->correlationId);
    }
  }
  for(size_t i=0;i<total;++i){
    const auto& b=completed[i];streams.insert({b.context,b.stream});CUpti_Activity* a=nullptr;
    while(true){
      r=cuptiActivityGetNextRecord(b.data,b.valid,&a);
      if(r==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
      if(r!=CUPTI_SUCCESS){failure=r;break;}
      ++kind_counts[static_cast<unsigned>(a->kind)];
      if(a->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL||a->kind==CUPTI_ACTIVITY_KIND_KERNEL){
        auto* k=reinterpret_cast<CUpti_ActivityKernel9*>(a);++all_kernels;
        if(!k->start||k->end<=k->start){invalid_record("kernel",k->start,k->end,k->deviceId,k->contextId,k->streamId,0,k->deviceId,k->contextId);continue;}
        if(k->end<=left||k->start>=right)continue;
        std::string name=k->name?k->name:"UNKNOWN";
        auto it=names.find(name);unsigned id;
        if(it==names.end()){id=ordered.size();names[name]=id;ordered.push_back(name);}else id=it->second;
        std::fprintf(f,"kernel,%llu,%llu,%llu,%u,%u,%u,%u,%u,0,0,%u,,\n",
          (unsigned long long)k->start,(unsigned long long)k->end,(unsigned long long)k->graphNodeId,
          k->graphId,k->deviceId,k->contextId,k->streamId,k->correlationId,id);++exported;
      }else if(a->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
        auto* m=reinterpret_cast<CUpti_ActivityMemcpy6*>(a);
        if(!m->start||m->end<=m->start){invalid_record("memcpy",m->start,m->end,m->deviceId,m->contextId,m->streamId,m->bytes,m->deviceId,m->contextId);continue;}
        if(m->end<=left||m->start>=right)continue;
        std::fprintf(f,"memcpy,%llu,%llu,%llu,%u,%u,%u,%u,%u,%llu,%u,0,,\n",
          (unsigned long long)m->start,(unsigned long long)m->end,(unsigned long long)m->graphNodeId,
          m->graphId,m->deviceId,m->contextId,m->streamId,m->correlationId,(unsigned long long)m->bytes,m->copyKind);
      }else if(a->kind==CUPTI_ACTIVITY_KIND_GRAPH_TRACE){
        auto* g=reinterpret_cast<CUpti_ActivityGraphTrace2*>(a);++all_graphs;
        if(!g->start||g->end<=g->start||g->deviceId!=g->endDeviceId||g->contextId!=g->endContextId){invalid_record("graph",g->start,g->end,g->deviceId,g->contextId,g->streamId,0,g->endDeviceId,g->endContextId,g->graphId,g->correlationId);continue;}
        if(g->end<=left||g->start>=right)continue;
        std::fprintf(f,"graph,%llu,%llu,0,%u,%u,%u,%u,%u,0,0,0,%u,%u\n",
          (unsigned long long)g->start,(unsigned long long)g->end,g->graphId,g->deviceId,g->contextId,
          g->streamId,g->correlationId,g->deviceId,g->endDeviceId);++exported;
      }else if(a->kind==CUPTI_ACTIVITY_KIND_MEMCPY2){
        auto* m=reinterpret_cast<CUpti_ActivityMemcpyPtoP4*>(a);
        if(!m->start||m->end<=m->start){invalid_record("memcpy2",m->start,m->end,m->deviceId,m->contextId,m->streamId,m->bytes,m->dstDeviceId,m->contextId);continue;}
        if(m->end<=left||m->start>=right)continue;
        std::fprintf(f,"memcpy2,%llu,%llu,%llu,%u,%u,%u,%u,%u,%llu,%u,0,%u,%u\n",
          (unsigned long long)m->start,(unsigned long long)m->end,(unsigned long long)m->graphNodeId,
          m->graphId,m->deviceId,m->contextId,m->streamId,m->correlationId,(unsigned long long)m->bytes,
          m->copyKind,m->srcDeviceId,m->dstDeviceId);
      }else if(a->kind==CUPTI_ACTIVITY_KIND_MEMSET){
        auto* m=reinterpret_cast<CUpti_ActivityMemset4*>(a);
        if(!m->start||m->end<=m->start){invalid_record("memset",m->start,m->end,m->deviceId,m->contextId,m->streamId,m->bytes,m->deviceId,m->contextId);continue;}
        if(m->end<=left||m->start>=right)continue;
        std::fprintf(f,"memset,%llu,%llu,%llu,%u,%u,%u,%u,%u,%llu,0,0,,\n",
          (unsigned long long)m->start,(unsigned long long)m->end,(unsigned long long)m->graphNodeId,
          m->graphId,m->deviceId,m->contextId,m->streamId,m->correlationId,(unsigned long long)m->bytes);
      }else if(a->kind==CUPTI_ACTIVITY_KIND_DRIVER||a->kind==CUPTI_ACTIVITY_KIND_RUNTIME||a->kind==CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API){
        auto* api=reinterpret_cast<CUpti_ActivityAPI*>(a);
        if(!graph_correlations.count(api->correlationId))continue;
        if(!api->start||api->end<=api->start){invalid_record("driver",api->start,api->end,UINT32_MAX,0,0,0,0,0,0,api->correlationId);continue;}
        const char* kind=a->kind==CUPTI_ACTIVITY_KIND_DRIVER?"driver":a->kind==CUPTI_ACTIVITY_KIND_RUNTIME?"runtime":"internal_api";
        std::fprintf(f,"%s,%llu,%llu,0,0,%u,0,0,%u,0,%u,0,,,%u\n",kind,
          (unsigned long long)api->start,(unsigned long long)api->end,UINT32_MAX,
          api->correlationId,api->returnValue,api->cbid);
      }
    }
  }
  for(const auto& s:streams){size_t lost=0;r=cuptiActivityGetNumDroppedRecords(s.first,s.second,&lost);if(r!=CUPTI_SUCCESS)failure=r;dropped+=lost;}
  std::fclose(f);
  std::fclose(invalid_file);invalid_file=nullptr;
  std::string kind_path=std::string(path)+".kinds.csv";f=std::fopen(kind_path.c_str(),"w");
  if(!f)return -111;
  std::fprintf(f,"kind,count\n");
  for(auto item:kind_counts)std::fprintf(f,"%u,%llu\n",item.first,(unsigned long long)item.second);
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
extern "C" uint64_t activity_all_graphs(){return all_graphs;}
extern "C" uint64_t activity_requested_buffers(){return requested.load();}
extern "C" uint64_t activity_completed_buffers(){return finished.load();}
extern "C" uint64_t activity_reserved_host_bytes(){return capacity*block;}
