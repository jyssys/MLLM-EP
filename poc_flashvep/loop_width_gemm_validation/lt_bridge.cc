// Finite cuBLASLt algorithm search. No approximation or weight conversion.
#include <cublasLt.h>
#include <cstdint>
#include <cstring>
#include <vector>

struct Plan {
  cublasLtHandle_t handle=nullptr;
  cublasLtMatmulDesc_t operation=nullptr;
  cublasLtMatrixLayout_t weight=nullptr,input=nullptr,output=nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> choices;
  std::vector<int> origins;
};
static thread_local int last_error=0;
extern "C" int lt_last_error(){return last_error;}
extern "C" size_t lt_version(){return cublasLtGetVersion();}
extern "C" void lt_destroy(void* address){
  auto* p=static_cast<Plan*>(address);if(!p)return;
  if(p->output)cublasLtMatrixLayoutDestroy(p->output);
  if(p->input)cublasLtMatrixLayoutDestroy(p->input);
  if(p->weight)cublasLtMatrixLayoutDestroy(p->weight);
  if(p->operation)cublasLtMatmulDescDestroy(p->operation);
  if(p->handle)cublasLtDestroy(p->handle);
  delete p;
}
extern "C" void* lt_create(int m,int n,int k,size_t workspace,int maximum){
  auto* p=new Plan();cublasStatus_t code;
  auto check=[&](cublasStatus_t result){last_error=result;return result==CUBLAS_STATUS_SUCCESS;};
  if(!check(cublasLtCreate(&p->handle))) {lt_destroy(p);return nullptr;}
  if(!check(cublasLtMatmulDescCreate(&p->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F))) {lt_destroy(p);return nullptr;}
  cublasOperation_t trans=CUBLAS_OP_T;
  if(!check(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)))) {lt_destroy(p);return nullptr;}
  // Row-major Y=X*W.T equals column-major Y.T=W*X.T.
  if(!check(cublasLtMatrixLayoutCreate(&p->weight,CUDA_R_16BF,k,n,k)) ||
     !check(cublasLtMatrixLayoutCreate(&p->input,CUDA_R_16BF,k,m,k)) ||
     !check(cublasLtMatrixLayoutCreate(&p->output,CUDA_R_16BF,n,m,n))) {lt_destroy(p);return nullptr;}
  cublasLtMatmulPreference_t preference=nullptr;
  if(!check(cublasLtMatmulPreferenceCreate(&preference))){lt_destroy(p);return nullptr;}
  code=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));
  if(!check(code)){cublasLtMatmulPreferenceDestroy(preference);lt_destroy(p);return nullptr;}
  std::vector<cublasLtMatmulHeuristicResult_t> results(maximum);int count=0;
  code=cublasLtMatmulAlgoGetHeuristic(p->handle,p->operation,p->weight,p->input,p->output,p->output,
    preference,maximum,results.data(),&count);
  if(!check(code)){cublasLtMatmulPreferenceDestroy(preference);lt_destroy(p);return nullptr;}
  auto add=[&](const cublasLtMatmulAlgo_t& algo,int origin){
    if(int(p->choices.size())>=maximum)return;
    for(auto& previous:p->choices)if(!std::memcmp(&previous.algo,&algo,sizeof(algo)))return;
    cublasLtMatmulHeuristicResult_t validated{};
    if(cublasLtMatmulAlgoCheck(p->handle,p->operation,p->weight,p->input,p->output,p->output,&algo,&validated)!=CUBLAS_STATUS_SUCCESS)return;
    if(validated.workspaceSize>workspace)return;
    validated.algo=algo;p->choices.push_back(validated);p->origins.push_back(origin);
  };
  for(int i=0;i<count;++i)if(results[i].state==CUBLAS_STATUS_SUCCESS)add(results[i].algo,0);
  // A fast small-M heuristic can use split-K reduction and change rounding.
  // Test the same valid algorithms without that reduction.
  for(int i=0;i<count;++i)if(results[i].state==CUBLAS_STATUS_SUCCESS){
    auto algo=results[i].algo;uint32_t split=1,reduction=0;
    cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction));
    add(algo,1);
  }
  // Check canonical-256 heuristic algorithms on the real unpadded layout.
  // This does not execute dummy rows.
  if(m!=256){
    cublasLtMatrixLayout_t input256=nullptr,output256=nullptr;
    cublasLtMatrixLayoutCreate(&input256,CUDA_R_16BF,k,256,k);
    cublasLtMatrixLayoutCreate(&output256,CUDA_R_16BF,n,256,n);
    std::vector<cublasLtMatmulHeuristicResult_t> canonical(maximum);int found=0;
    code=cublasLtMatmulAlgoGetHeuristic(p->handle,p->operation,p->weight,input256,output256,output256,
        preference,maximum,canonical.data(),&found);
    if(code==CUBLAS_STATUS_SUCCESS)for(int i=0;i<found;++i)if(canonical[i].state==CUBLAS_STATUS_SUCCESS)add(canonical[i].algo,2);
    cublasLtMatrixLayoutDestroy(output256);cublasLtMatrixLayoutDestroy(input256);
  }
  cublasLtMatmulPreferenceDestroy(preference);
  return p;
}
extern "C" int lt_count(void* p){return static_cast<Plan*>(p)->choices.size();}
extern "C" int lt_origin(void* p,int index){return static_cast<Plan*>(p)->origins[index];}
extern "C" int lt_attribute(void* address,int index,int attribute){
  auto* p=static_cast<Plan*>(address);int value=-1;size_t written=0;
  if(index<0 || index>=int(p->choices.size()))return -999;
  cublasLtMatmulAlgoConfigGetAttribute(&p->choices[index].algo,
    static_cast<cublasLtMatmulAlgoConfigAttributes_t>(attribute),&value,sizeof(value),&written);
  return value;
}
extern "C" size_t lt_workspace(void* address,int index){return static_cast<Plan*>(address)->choices[index].workspaceSize;}
extern "C" int lt_run(void* address,int index,const void* x,const void* w,void* y,void* workspace,size_t bytes,void* stream){
  auto* p=static_cast<Plan*>(address);
  if(index<0 || index>=int(p->choices.size()))return -20;
  float alpha=1.f,beta=0.f;
  return cublasLtMatmul(p->handle,p->operation,&alpha,w,p->weight,x,p->input,
    &beta,y,p->output,y,p->output,&p->choices[index].algo,workspace,bytes,
    reinterpret_cast<cudaStream_t>(stream));
}
