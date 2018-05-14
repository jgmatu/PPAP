#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

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
