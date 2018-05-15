#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <cuda.h>

#define THREADS 32

#define WIDTH 4
#define HEIGHT 2

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

      dest[idy * WIDTH + idx] = src[idy * WIDTH + idx]; // Copio tal cual con los mismos indices facil... :)
}

__global__ void copy_shared(int *src, int *dst)
{
      __shared__ int mem[THREADS][THREADS + 1];

      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      if (idx >= WIDTH || idy >= HEIGHT) return;

      mem[threadIdx.x][threadIdx.y] = src[idy * WIDTH + idx]; // Añado el valor en la memoria compartida...
      __syncthreads();

      dst[idy * WIDTH + idx] = mem[threadIdx.x][threadIdx.y]; // Añado en su posicion natural el valor de la memoria
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

      __shared__ int mem[THREADS][THREADS + 1];
      mem[threadIdx.x][threadIdx.y] = src[idx * HEIGHT + idy]; // Hago las posiciones traspuestas
                                                              // en la memoria compartida...
      __syncthreads();
      dst[idy * WIDTH + idx] = mem[threadIdx.x][threadIdx.y]; // Añado en su posicion natural el valor de la shared
                                                               // que tiene el valor de la traspuesta....
}

__global__ void matrixAddPitch (int *a, int *b, int*c, int pitch) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idx > pitch || idy > HEIGHT) return;

	c[idy * pitch + idx] = a[idy * pitch + idx] + b[idy * pitch + idx];
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

void print_matrix(const int *matrix, const int w, const int h) {
      fprintf(stdout, "%s\n", "Print matrix...");
      for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; ++j) {
                  fprintf(stdout, "%5d", matrix[i * w + j]);
            }
            fprintf(stdout, "%s\n", "");
      }
      fprintf(stdout, "%s\n", "");
}

void addPitch()
{
      int n = WIDTH * HEIGHT;

      dim3 t (16, 16);
      dim3 b ( (WIDTH - 1) / t.x + 1, (HEIGHT - 1) / t.y  + 1);
      int *h_a, *h_b, *h_c;
      int *d_a, *d_b, *d_c;
      int size = sizeof(int) * n;
      size_t pitch;

      h_a = (int *) malloc (size);
      h_b = (int *) malloc (size);
      h_c = (int *) malloc (size);

      for (int i = 0; i < n; i++) {
            h_a[i] = i;
            h_b[i] = i;
      }

      cudaMallocPitch(&d_a, &pitch, WIDTH * sizeof(int), HEIGHT);
      cudaMallocPitch(&d_b, &pitch, WIDTH * sizeof(int), HEIGHT);
      cudaMallocPitch(&d_c, &pitch, WIDTH * sizeof(int), HEIGHT);

      cudaMemcpy2D (d_a, pitch, h_a, WIDTH * sizeof(int), WIDTH * sizeof(int), HEIGHT, cudaMemcpyHostToDevice);
      cudaMemcpy2D (d_b, pitch, h_b, WIDTH * sizeof(int), WIDTH * sizeof(int), HEIGHT, cudaMemcpyHostToDevice);

      matrixAddPitch <<<b, t>>> (d_a, d_b, d_c, pitch / sizeof(int));
      cudaMemcpy2D (h_c, WIDTH * sizeof(int), d_c, pitch, WIDTH * sizeof(int), HEIGHT, cudaMemcpyDeviceToHost);

      print_matrix(h_c, HEIGHT, WIDTH);

      free(h_a);
      free(h_b);
      free(h_c);

      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
}
void traspose() {
      int *matrix = NULL;
      int *dev_matrix = NULL;
      int *dev_traspose = NULL;

      mi_malloc_int(&matrix, WIDTH * HEIGHT);
      init(matrix);
      print_matrix(matrix, WIDTH, HEIGHT);

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
      traspose<<<b, t>>>(dev_matrix, dev_traspose);
      // Call kernel << b ,  t >> (a , b);
      cudaMemcpy(matrix, dev_traspose, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
      // dest, src, size, cudaMemcpyDeviceToHost;
      fprintf(stdout, "Time : %lf ms\n", (get_time() - now)  / 1000000.0f);
      // ... END PARARELL CODE ...

      print_matrix(matrix, HEIGHT, WIDTH);
      fprintf(stdout, "Num Blocks (x:%d y:%d)\n", b.x, b.y);
}

int main(int argc, char const *argv[])
{
      traspose();
      return 0;
}
