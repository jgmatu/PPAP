#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <omp.h>

const unsigned long WIDTH = 8192;
const unsigned long HEIGHT = 8192;
const unsigned long NTHREADS = 4;

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
      for (unsigned long i = 0; i < HEIGHT; i++) {
            for (unsigned long j = 0; j < WIDTH; ++j) {
                  h_v[i * HEIGHT + j] = numb;
            }
      }
}


void add(int *a, int *b, int *result)
{
      #pragma omp parallel for num_threads(4)
      for (unsigned long i = 0; i < HEIGHT; ++i) {
            for (unsigned long j = 0; j < WIDTH ; ++j) {
                  result[i * HEIGHT + j] = a[i * HEIGHT + j] + b[i * HEIGHT + j];
            }
      }
}

int main( void ) {
      unsigned long now = get_time();

      int *result, *h_a, *h_b;
      unsigned long size = WIDTH * HEIGHT * sizeof(int);

      result = (int*) malloc( size );
      h_a = (int*) malloc( size );
      h_b = (int*) malloc( size );

      if (!h_a || !h_b || !result) {
            fprintf(stderr, "Error allocate memory... %s\n", strerror(errno));
            exit(1);
      }

      init(h_a, 7);
      init(h_b, 2);
      add(h_a, h_b, result);
      free(h_a), free(h_b), free(result);

      fprintf(stderr, "Time %lu\n", get_time() - now);
      exit(0);
}
