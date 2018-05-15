#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <omp.h>

const int NTHREADS = 4;
static const long n = 1000000000;

unsigned long get_time()
{
      struct timespec ts;

      if (clock_gettime(0, &ts) < 0) {
            fprintf(stderr, "Error calc time... %s\n", strerror(errno));
            exit(1);
      }
      return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

double getPI() {
      double sum = 0.0;
      double step = 1.0 / (double) n;

      #pragma omp parallel for reduction (+:sum) num_threads(NTHREADS)
      for (int i = 0; i < n; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
      }
      return step * sum;
}

void showTimesOpenMP ()
{
      double total = 0.0;
      unsigned N = 10;
      unsigned long before = get_time();

      #pragma omp parallel for reduction (+:total) num_threads(NTHREADS)
      for (unsigned i = 0; i < N; ++i) {
            fprintf(stderr, "%2.20lf\n", getPI());
            total += (get_time() - before) / 1000000.0;
      }
      fprintf(stderr, "Max Threads... %d\n", omp_get_max_threads());
      fprintf(stderr, "Avg time %lfms\n", total / N);
}

int main(int argc, char *argv[])
{
      showTimesOpenMP();

//      fprintf(stderr, "%d\n", omp_num_procs());
}
