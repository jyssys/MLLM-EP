#include <cuda_runtime.h>
#include <stdint.h>

__device__ __constant__ uint32_t constants[64]={
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ uint32_t rotate(uint32_t x,int n){return (x>>n)|(x<<(32-n));}

__global__ void hash_leaves(const unsigned char* input,uint64_t bytes,uint64_t chunk,unsigned char* output){
  uint64_t leaf=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
  uint64_t begin=leaf*chunk;if(begin>=bytes)return;
  uint64_t length=min(chunk,bytes-begin),padded=((length+9+63)/64)*64;
  uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  for(uint64_t offset=0;offset<padded;offset+=64){
    uint32_t w[16];
    #pragma unroll
    for(int i=0;i<16;i++){
      uint32_t word=0;
      #pragma unroll
      for(int j=0;j<4;j++){
        uint64_t p=offset+i*4+j;unsigned char byte=0;
        if(p<length)byte=input[begin+p];
        else if(p==length)byte=0x80;
        else if(p>=padded-8)byte=(unsigned char)((length*8)>>(8*(padded-1-p)));
        word=(word<<8)|byte;
      }
      w[i]=word;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
    #pragma unroll
    for(int i=0;i<64;i++){
      if(i>=16){
        uint32_t x=w[(i-15)&15],y=w[(i-2)&15];
        uint32_t s0=rotate(x,7)^rotate(x,18)^(x>>3);
        uint32_t s1=rotate(y,17)^rotate(y,19)^(y>>10);
        w[i&15]+=s0+w[(i-7)&15]+s1;
      }
      uint32_t s1=rotate(e,6)^rotate(e,11)^rotate(e,25);
      uint32_t choose=(e&f)^((~e)&g);
      uint32_t t1=z+s1+choose+constants[i]+w[i&15];
      uint32_t s0=rotate(a,2)^rotate(a,13)^rotate(a,22);
      uint32_t majority=(a&b)^(a&c)^(b&c),t2=s0+majority;
      z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
  }
  #pragma unroll
  for(int i=0;i<8;i++){
    output[leaf*32+i*4]=(unsigned char)(h[i]>>24);
    output[leaf*32+i*4+1]=(unsigned char)(h[i]>>16);
    output[leaf*32+i*4+2]=(unsigned char)(h[i]>>8);
    output[leaf*32+i*4+3]=(unsigned char)h[i];
  }
}

extern "C" int sha256_leaves(const void* input,uint64_t bytes,uint64_t chunk,void* output,void* stream){
  if(!bytes||!chunk||chunk>65536)return (int)cudaErrorInvalidValue;
  uint64_t leaves=(bytes+chunk-1)/chunk;
  hash_leaves<<<(leaves+127)/128,128,0,(cudaStream_t)stream>>>((const unsigned char*)input,bytes,chunk,(unsigned char*)output);
  return (int)cudaGetLastError();
}
