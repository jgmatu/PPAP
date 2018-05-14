#include <iostream> //cout, endl
#include <stdlib.h> //atoi
#include <stdio.h> //getchar
#include <omp.h> //omp

void func() {
      #pragma omp parallel num_threads(4)
      {
            fprintf(stdout, "%s\n", "in the parallel region inside the subprogram.");
      }
      fprintf(stdout, "%s\n", "in the subprogram outside the parallel region,");
}

int main(int argc, char **argv)
{
      fprintf(stdout, "%s\n", "in the main program outside the parallel region,");

      #pragma omp parallel num_threads(4)
      {
            func();
            fprintf(stdout, "%s\n", "in the parallel region in the main program");
      }
      return 0;
}
