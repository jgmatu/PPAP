#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <cuda.h>

#define THREADS 32

#define WIDTH 8192
#define HEIGHT 8192

// See times between copy and traspose and realize the same operation
// with shared memory.

// Traspose is slower than copy...
// Traspose is need to shared memory to improve the performance...

// https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/

__global__ void copy(int *src, int *dest)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      if (idx >= WIDTH || idy >= HEIGHT) return;

      dest[idx * HEIGHT + idy] = src[idx * HEIGHT + idy]; // Copio tal cual con los mismos indices facil... :)
}

__global__ void copy_shared(int *src, int *dst)
{
      __shared__ int mem[THREADS][THREADS];

      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      if (idx >= WIDTH || idy >= HEIGHT) return;

      mem[threadIdx.x][threadIdx.y] = src[idx * HEIGHT + idy]; // Añado el valor en la memoria compartida...
      __syncthreads();

      dst[idx * HEIGHT + idy] = mem[threadIdx.x][threadIdx.y]; // Añado en su posicion natural el valor de la memoria
                                                              // compartida...
}

__global__ void traspose(int *src, int *dest)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      if (idx >= WIDTH || idy >= HEIGHT) return;

      dest[idx * HEIGHT + idy] = src[idy * WIDTH + idx]; // Cambio el valor de la matriz a la traspuesta
                                                         // con los índices de acceso a la matriz...
}

__global__ void traspose_shared(int *src, int *dst)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      if (idx >= WIDTH || idy >= HEIGHT) return;

      __shared__ int mem[THREADS][THREADS];
      mem[threadIdx.x][threadIdx.y] = src[idy * WIDTH + idx]; // Hago las posiciones traspuestas
                                                              // en la memoria compartida...
      __syncthreads();
      dst[idx * HEIGHT + idy] = mem[threadIdx.x][threadIdx.y]; // Añado en su posicion natural el valor de la shared
                                                               // que tiene el valor de la traspuesta....
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

void mi_malloc_int(int **i, int size) {
      *i = (int *) malloc( sizeof(int) * size);
      if (*i == NULL) {
            fprintf(stderr, "Error malloc %s\n", strerror(errno));
            exit(1);
      }
      memset(*i, 0, sizeof(int) * size);
}

void init(int *h_v) {
      for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; ++j) {
                  h_v[i * WIDTH + j] = i * WIDTH + j;
            }
      }
}

void print_matrix(const int *matrix) {
      fprintf(stdout, "%s\n", "Print matrix...");
      for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; ++j) {
                  fprintf(stdout, "%5d", matrix[i * WIDTH + j]);
            }
            fprintf(stdout, "%s\n", "");
      }
      fprintf(stdout, "%s\n", "");
}

void print_traspose(const int *traspose) {
      fprintf(stdout, "%s\n", "Print traspose...");
      for (int i = 0; i < WIDTH; i++) {
            for (int j = 0; j < HEIGHT; ++j) {
                  fprintf(stdout, "%5d", traspose[i * HEIGHT + j]);
            }
            fprintf(stdout, "%s\n", "");
      }
      fprintf(stdout, "%s\n", "");
}

int main(int argc, char const *argv[])
{
      int *matrix = NULL;
      int *dev_matrix = NULL;
      int *dev_traspose = NULL;

      mi_malloc_int(&matrix, WIDTH * HEIGHT);
      init(matrix);
//      print_matrix(matrix);

      cudaMalloc(&dev_matrix, sizeof(int) * WIDTH * HEIGHT);
      // &dst, size...

      cudaMemcpy(dev_matrix, matrix, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
      // dst, src, size, cudaMemcpyHostToDevice...;

      cudaMalloc(&dev_traspose, sizeof(int) * WIDTH * HEIGHT);
      // &dst, size...

      cudaMemset(dev_traspose, 0, sizeof(int) * WIDTH * HEIGHT);
      // dst, value byte 0, size...

      dim3 t(THREADS, THREADS);
      dim3 b((WIDTH - 1) / t.x + 1, (HEIGHT - 1) / t.y + 1);

      // ... START PARARELL CODE ...
      unsigned long now = get_time();
      copy<<<b, t>>>(dev_matrix, dev_traspose);
      // Call kernet << b ,  t >> (a , b);
      cudaMemcpy(matrix, dev_traspose, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
      // dest, src, size, cudaMemcpyDeviceToHost;
      fprintf(stdout, "Time : %lf ms\n", (get_time() - now)  / 1000000.0f);
      // ... END PARARELL CODE ...
      
//      print_traspose(matrix);

      fprintf(stdout, "Num Blocks (x:%d y:%d)\n", b.x, b.y);
      return 0;
}
