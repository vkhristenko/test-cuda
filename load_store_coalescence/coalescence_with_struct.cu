/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <assert.h>

template<typename T>
struct Point {
    T x;
//    T y[1];
};

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

template <typename T>
__global__ 
void compute(Point<T>* a)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  a[idx].x = a[idx].x + 1;
}

template<typename T>
__global__
void compute(T* a) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    a[idx] = a[idx] + 1;
}

template <typename T>
void runTestSimple(int deviceId, int n)
{
  int blockSize = 256;
  float ms;

  Point<T> *d_a;
  cudaEvent_t startEvent, stopEvent;
    
//  int n = nMB*1024*1024/sizeof(Point<T>);
  
  int nEffectiveMB = n * sizeof(T) / (1024*1024);

  // NB:  d_a(33*nMB) for stride case
  checkCuda( cudaMalloc(&d_a, n * sizeof(T)) );

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  printf("Offset, Bandwidth (GB/s):\n");
  
  compute<<<n/blockSize, blockSize>>>(d_a); // warm up

  for (int i = 0; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    compute<<<n/blockSize, blockSize>>>(d_a);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nEffectiveMB/ms);
  }

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  cudaFree(d_a);
}

template <typename T>
void runTest(int deviceId, int n)
{
  int blockSize = 256;
  float ms;

  Point<T> *d_a;
  cudaEvent_t startEvent, stopEvent;
    
//  int n = nMB*1024*1024/sizeof(Point<T>);
  int nEffectiveMB = n * sizeof(T) / (1024*1024);

  // NB:  d_a(33*nMB) for stride case
  checkCuda( cudaMalloc(&d_a, n * sizeof(Point<T>)) );

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  printf("Offset, Bandwidth (GB/s):\n");
  
  compute<<<n/blockSize, blockSize>>>(d_a); // warm up

  for (int i = 0; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(Point<T>)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    compute<<<n/blockSize, blockSize>>>(d_a);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nEffectiveMB/ms);
  }

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  cudaFree(d_a);
}

int main(int argc, char **argv)
{
  int n = 1000000;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {    
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }
  
  cudaDeviceProp prop;
  
  checkCuda( cudaSetDevice(deviceId) )
  ;
  checkCuda( cudaGetDeviceProperties(&prop, deviceId) );
  printf("Device: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", n);
  
  printf("%s Precision\n", bFp64 ? "Double" : "Single");
  
  if (bFp64) runTest<double>(deviceId, n);
  else       runTest<float>(deviceId, n);
  if (bFp64) runTestSimple<double>(deviceId, n);
  else       runTestSimple<float>(deviceId, n);
}
