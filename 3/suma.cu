#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <cuda.h>


const unsigned long WIDTH = 8192;
const unsigned long HEIGHT = 8192;
#define THREADS 32

__global__ void add(int* a, int* b, int* c)
{
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      int idy = threadIdx.y + blockIdx.y * blockDim.y;

      if (idx > WIDTH || idy > HEIGHT) return;

      c[idy * WIDTH + idx] = a[idy * WIDTH + idx] + b[idy * WIDTH + idx];
}

unsigned long get_time()
{
      struct timespec ts;

      if (clock_gettime(0, &ts) < 0) {
            fprintf(stderr, "Error calc time... %s\n", strerror(errno));
            exit(1);
      }
      return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

void  init(int* h_v, int numb) {
      for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; ++j) {
                  h_v[i * HEIGHT + j] = numb;
            }
      }
}

void print_results(const int *result)
{
      fprintf(stderr, "%s\n", "Result...");
      for (int i = 0; i < HEIGHT; ++i) {
            for (int j = 0; j < WIDTH; ++j) {
                  fprintf(stderr, " %d ", result[i * HEIGHT + j]);
            }
            fprintf(stderr, "%s\n", "");
      }
      fprintf(stderr, "%s\n", "");
}

int main( void ) {
      unsigned long now = get_time();

      int *result, *h_a, *h_b;

      int *dev_a, *dev_b, *dev_c;
      int size = WIDTH * HEIGHT * sizeof(int);

      result = (int*) malloc( size );
      h_a = (int*) malloc( size );
      h_b = (int*) malloc( size );

      init(h_a, 7);
      init(h_b, 2);

      cudaMalloc( &dev_a, size );
      cudaMalloc( &dev_b, size );
      cudaMalloc( &dev_c, size );

      // se transfieren los datos a memoria de dispositivo.
      cudaMemcpy( dev_a, h_a, size, cudaMemcpyHostToDevice);
      cudaMemcpy( dev_b, h_b, size, cudaMemcpyHostToDevice);
      cudaMemset(dev_c, 0, size);

      dim3 th(THREADS, THREADS);
      dim3 blocks((WIDTH + th.x - 1) / th.x , (HEIGHT + th.y - 1) / th.y);
      add<<<blocks, th>>>(dev_a, dev_b, dev_c);


      // se transfieren los datos del dispositivo a memoria.
      cudaMemcpy(result, dev_c, size, cudaMemcpyDeviceToHost);
      free(h_a), free(h_b), free(result);
      cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

      fprintf(stderr, "Time %lu\n", get_time() - now);
      return 0;
}
