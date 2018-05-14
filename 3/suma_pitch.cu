#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define BLOCKS 8
#define THREADS 16
#define WIDTH 128
#define HEIGHT 64

__global__ void add(int* a, int* b, int* c)
{
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      int idy = threadIdx.y + blockIdx.y * blockDim.y;

      if (idx > WIDTH || idy > HEIGHT) return;

      c[idy * WIDTH + idx] = a[idy * WIDTH + idx] + b[idy * WIDTH + idx];
}

void  init(int* h_v, int numb) {
      for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; ++j) {
                  h_v[i * HEIGHT + j] = numb;
            }
      }
}

// N
int main( void ) {
      int *result, *h_a, *h_b;

      int *dev_a, *dev_b, *dev_c;
      int size = HEIGHT * WIDTH * sizeof(int);
      size_t size_pitch;

      result = (int*) malloc( size );
      h_a = (int*) malloc( size );
      h_b = (int*) malloc( size );

      init(h_a, 1);
      init(h_b, 2);

      // cudaMallocPitch cudaMemcpy2D
      cudaError_t error = cudaMallocPitch( &dev_a, &size_pitch, WIDTH * sizeof(int), HEIGHT );
      // Warning: el pitch varia por el tamaño que reservas si a b c fueren diferente tamaño..
      error = cudaMallocPitch( &dev_b, &size_pitch, WIDTH * sizeof(int), HEIGHT );
      error = cudaMallocPitch( &dev_c, &size_pitch, WIDTH * sizeof(int), HEIGHT );

//      cudaError_t error = cudaMalloc( &dev_a, WIDTH * sizeof(int) * HEIGHT );
//      error = cudaMalloc( &dev_b, size );
//      error = cudaMalloc( &dev_c, size );

//      fprintf(stderr, "Size %lu\n", WIDTH * sizeof(int) * HEIGHT);
      // se transfieren los datos a memoria de dispositivo.
//      cudaMemcpy( dev_a, h_a, size , cudaMemcpyHostToDevice);
//      cudaMemcpy( dev_b, h_b, size , cudaMemcpyHostToDevice);
//      cudaMemset(dev_c, 0, size);

// cudaMemcpy2D (void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
      error = cudaMemcpy2D( dev_a, size_pitch, h_a, WIDTH * sizeof(int), WIDTH * sizeof(int), HEIGHT, cudaMemcpyHostToDevice);
      error = cudaMemcpy2D( dev_b, size_pitch, h_b, WIDTH * sizeof(int), WIDTH * sizeof(int), HEIGHT, cudaMemcpyHostToDevice);

// cudaError_t cudaMemset2D (void *devPtr, size_t pitch, int value, size_t width, size_t height)
      error = cudaMemset2D(dev_c, size_pitch, 0, sizeof(int) * WIDTH, HEIGHT);

      dim3 t(THREADS, THREADS);
      dim3 b( (WIDTH + t.x - 1) / t.x , (HEIGHT + t.y - 1) / t.y);
      add<<<b, t>>>( dev_a, dev_b, dev_c);

      // se transfieren los datos del dispositivo a memoria.
//    cudaMemcpy(result, dev_c, size, cudaMemcpyDeviceToHost);
//    cudaMemcpy2D (void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
      cudaMemcpy2D(result, WIDTH * sizeof(int), dev_c, size_pitch, WIDTH * sizeof(int), HEIGHT, cudaMemcpyDeviceToHost);

      fprintf(stderr, "%s\n", "Result...");
      for (int i = 0; i < HEIGHT; ++i) {
            for (int j = 0; j < WIDTH; ++j) {
                  fprintf(stderr, " %d ", result[i * HEIGHT + j]);
            }
            fprintf(stderr, "%s\n", "");
      }
      fprintf(stderr, "%s\n", "");

      free(h_a), free(h_b), free(result);
      cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
      return 0;
}
