#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define THREADS 16
#define BLOCKS 2

__global__ void add(int *array) {
      int temp = 0;

      int before = (blockIdx.x * blockDim.x + threadIdx.x + 1) % (THREADS * BLOCKS);
      int after = (blockIdx.x * blockDim.x + threadIdx.x - 1) % (THREADS * BLOCKS);

      temp += array[blockIdx.x * blockDim.x + threadIdx.x];
      temp += array[before];
      temp += array[after];
      __syncthreads(); //evita condición de carrera...

      array[blockIdx.x * blockDim.x + threadIdx.x] = temp;
}

void  init(int* h_v, int numb) {
      for (int i = 0; i < THREADS * BLOCKS; i++) {
            h_v[i] = numb;
      }
}

int main( void ) {
      int *result, *h_a;

      int *dev_a;
      int size = THREADS * BLOCKS * sizeof(int);

      result = (int*) malloc( size );
      h_a = (int*) malloc( size );

      memset(result, 0, size);
      memset(h_a, 0, size);

      init(h_a, 1);

      cudaMalloc(&dev_a, size);

      // se transfieren los datos a memoria de dispositivo...
      cudaMemcpy(dev_a, h_a, size, cudaMemcpyHostToDevice);

      add<<<BLOCKS, THREADS>>>(dev_a);

      // se transfieren los datos del dispositivo a memoria.
      cudaMemcpy(result, dev_a, size, cudaMemcpyDeviceToHost);

      for (int i = 0; i < THREADS * BLOCKS; i++) {
            fprintf(stderr, "Result : %d\n", result[i]);
      }

      free(h_a), free(result);
      cudaFree(dev_a);
      return 0;
}
