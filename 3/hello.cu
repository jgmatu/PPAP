#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

// int pos = threadIdx.x + blockIdx.x * blockDim.x;

__global__ void kernelVacio( void ) {
      if (threadIdx.x < 10) {
            printf("Data: %s Id Thread: %d Id block : %d Num threads block : %d\n", "helloWorld!",
                  threadIdx.x, blockIdx.x, blockDim.x);
      }
}

int main( void ) {
      kernelVacio<<<8,32>>>(); //kernel launch con grid mínimo
      cudaDeviceSynchronize();
      return 0;
}
