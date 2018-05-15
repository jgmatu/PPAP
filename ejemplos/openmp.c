#include <iostream> //cout, endl
#include <stdlib.h> //atoi
#include <stdio.h> //getchar
#include <string.h>
#include <errno.h>
#include <omp.h> //omp


unsigned long get_time()
{
      struct timespec ts;

      if (clock_gettime(0, &ts) < 0) {
            fprintf(stderr, "Error calc time... %s\n", strerror(errno));
            exit(1);
      }
      return ts.tv_sec * 1000000000L + ts.tv_nsec;
}


void func() {
      #pragma omp parallel num_threads(4)
      {
            fprintf(stdout, "%s\n", "in the parallel region inside the subprogram.");
      }
      fprintf(stdout, "%s\n", "in the subprogram outside the parallel region,");
}

const long N = 100000;

int main(int argc, char **argv)
{
      fprintf(stdout, "%s\n", "in the main program outside the parallel region,");

      #pragma omp parallel num_threads(4)
      {
            func();
            fprintf(stdout, "%s\n", "in the parallel region in the main program");
      }

      int tsum = 0;
      #pragma omp parallel
      {
            tsum += omp_get_thread_num();/* the thread number */
      }
      printf("Sum is %d\n", tsum);

      int data[N][2];
      for (int i = 0; i < N; i++) {
            data[i][0] = 1;
      }

      unsigned long now = get_time();
      int sum = 0;
      #pragma omp parallel for reduction (+:sum)
      for (unsigned long i = 0; i < N; i++) {
            sum += data[omp_get_thread_num()][0];
      }
      fprintf(stdout, "Time %lf ms\n", (get_time() - now) / 1000000.0);
      fprintf(stdout, "Result %d\n", sum);
      return 0;
}
