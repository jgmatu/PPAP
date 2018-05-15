#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
/*
#define N 512
__global__ void dotProd( int *a, int *b, int *c ) {
      __shared__ int temp[N];

      temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

      __syncthreads(); // Evita condición de carrera.
      if( 0 == threadIdx.x ) {
            int sum = 0;
            for(int i = 0; i < N; i++ ) {
                  sum += temp[i]; //lento
            }
            *c = sum;
      }
}

#define N 2048
#define THREADS_PER_BLOCK 512

__global__ void dotProd( int *a, int *b, int *c ) {
      __shared__ int temp[THREADS_PER_BLOCK];
      int index = threadIdx.x + blockIdx.x * blockDim.x;

      temp[threadIdx.x] = a[index] * b[index];
      __syncthreads(); // Hasta que no rellenen todos los thread temp no puedo continuar...

      if(threadIdx.x == 0) {
            int sum = 0;
            for( int i= 0; i < THREADS_PER_BLOCK; i++ ) {
                  sum += temp[i];
            }
            c[blockIdx.x] = sum;
      }
}
*/
const int THREADS_PER_BLOCK = 32;
const int N = 2048;

__global__ void mult(int *a, int *b, int *c)
{
      int pos = threadIdx.x + blockDim.x * blockIdx.x;
      if (pos >= N) return;

      c[pos] = a[pos] * b[pos];
}

__global__ void shared_mult(int *a, int *b, int *c)
{
      __shared__ int mem[THREADS_PER_BLOCK];
      int pos = threadIdx.x + blockIdx.x * blockDim.x;
      mem[threadIdx.x] = a[pos]  * b[pos];

      __syncthreads();
      c[pos] = mem[threadIdx.x];
}

int main(int argc, char const *argv[]) {
      int *a, *b, *c;
      int *dev_a, *dev_b, *dev_c;
      int size = sizeof(int) * N;

      a = (int *) malloc(size);
      b = (int *) malloc(size);
      c = (int *) malloc(size);

      for (int i = 0; i < N ; i++) {
            a[i] = b[i] = 3;
      }

      cudaMalloc(&dev_a, size);
      cudaMalloc(&dev_b, size);
      cudaMalloc(&dev_c, size);

      cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
      cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
      cudaMemset(dev_c, 0, size);

      shared_mult<<<(N - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c);

      cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

      for (int i = 0; i < N ; i++) {
            fprintf(stdout, "Numb : %d\n", c[i]);
      }
      cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
      return 0;
}
