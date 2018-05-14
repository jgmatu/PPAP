#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define SIZE 10000

unsigned long get_time()
{
      struct timespec ts;

      if (clock_gettime(0, &ts) < 0) {
            fprintf(stderr, "Error calc time... %s\n", strerror(errno));
            exit(1);
      }
      return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

void addVector(const float *a, const float *b, float *c, int n)
{
      #pragma omp parallel for
      for (unsigned i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
      }
}

void addVector_unsigned(const unsigned *a, const unsigned *b, unsigned *c, int n)
{
      #pragma omp parallel for
      for (unsigned i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
      }
}

void addVector_char(const char *a, const char *b, char *c, int n)
{
      for (unsigned i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
      }
}

void addMatrix(const float *a, const float *b, float *c, int rows, int cols) {
      for (unsigned i = 0; i < rows; ++i) {
            for (unsigned j = 0; j < cols; ++j) {
                  c[j * cols + j] = a[i * cols + j] + b[i * cols + j];
            }
      }
}

void initVector(float *a, float numb, int n) {
      for (unsigned i = 0; i < n; ++i) {
            a[i] = numb;
      }
}

void initVector_char(char *a, char numb, int n) {
      for (unsigned i = 0; i < n; ++i) {
            a[i] = numb;
      }
}

void initMatrix_float(float* a, float numb, int rows, int cols) {
      for (unsigned i = 0; i < rows; ++i) {
            for (unsigned j = 0; j < cols; ++j) {
                  a[i * cols + j] = numb;
            }
      }
}

bool checkVectors(const float *a, const float *b, int size) {
      for (unsigned i = 0; i < size; ++i) {
            if (abs(a[i] - b[i]) > 0.00001f) {
                  return false;
            }
      }
      return true;
}


void initMatrix(int m[SIZE][SIZE], int numb, int rows, int cols) {
      for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                  m[i][j] = numb;
            }
      }
}

void addMatrixs(const int m[SIZE][SIZE], const int d[SIZE][SIZE], int r[SIZE][SIZE], int rows, int cols) {
      for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                  r[i][j] = m[i][j] + d[i][j];
            }
      }
}


void mi_malloc(int **i, int size) {
      *i = malloc( sizeof(int) * size);
      if (*i == NULL) {
            fprintf(stderr, "Error malloc %s\n", strerror(errno));
      }
      memset(*i, 0, sizeof(int) * size);
}

void mi_malloc_float(float **i, int size) {
      *i = malloc( sizeof(float) * size);
      if (*i == NULL) {
            fprintf(stderr, "Error malloc %s\n", strerror(errno));
      }
      memset(*i, 0, sizeof(float) * size);
}

void mi_malloc_char(char **i, int size) {
      *i = malloc( sizeof(char) * size);
      if (*i == NULL) {
            fprintf(stderr, "Error malloc %s\n", strerror(errno));
      }
      memset(*i, 0, sizeof(char) * size);
}


void randomMatrix(int* m, int size) {
      srand(time(0));

      for (unsigned i = 0; i < size; ++i) {
            for (unsigned j = 0; j < size; ++j) {
                  m[i * size + j] = rand() % 10;
            }
      }
}

void print_vector(const float *a, int n) {
      for (unsigned i = 0; i < n; ++i) {
            fprintf(stdout, "%f\n", a[i]);
      }
}

void print_matrix(const float *a , int rows, int cols) {
      for (unsigned i = 0; i < rows; ++i) {
            for (unsigned j = 0; j < cols; ++j) {
                  fprintf(stderr, "%f\n", a[i * cols + j]);
            }
      }
}

float sum(const float* a, int n) {
      float total = 0.0;

      #pragma omp parallel for reduction (+:total)
      for (unsigned i = 0; i < n; ++i) {
            total += a[i];
      }
      return total;
}

float max(const float* a, int n) {
      float max = FLT_MIN;

      #pragma omp parallel for reduction(max : max)
      for (unsigned i = 0; i < n; ++i) {
            max = fmaxf(max, a[i]);
      }
      return max;
}

float min(const float* a, int n) {
      float min = FLT_MAX;

      for (unsigned i = 0; i < n; ++i) {
            min = fminf(min, a[i]);
      }
      return min;
}

int main(int argc, char const *argv[]) {

/*
      fprintf(stderr, "bool %ld\n", sizeof(bool));
      fprintf(stderr, "char %ld\n", sizeof(char));
      fprintf(stderr, "short %ld\n", sizeof(short));
      fprintf(stderr, "unsigned short %ld\n", sizeof(unsigned short));
      fprintf(stderr, "int %ld\n", sizeof(int));
      fprintf(stderr, "unsigned int %ld\n", sizeof(unsigned));
      fprintf(stderr, "long %ld\n", sizeof(long));
      fprintf(stderr, "unsigned long %ld\n", sizeof(unsigned long));
      fprintf(stderr, "long long %ld\n", sizeof(long long));
      fprintf(stderr, "unsigned long long %ld\n", sizeof(unsigned long long));
      fprintf(stderr, "float %ld\n", sizeof(float));
      fprintf(stderr, "double %ld\n", sizeof(double));
      fprintf(stderr, "long double %ld\n", sizeof(long double));
*/
/*
      int i, *p;
      i = 2;
      p = &i;

      fprintf(stderr, "%p\n", &p);
      fprintf(stderr, "%p\n", p);
      fprintf(stderr, "%p\n", &i);
      fprintf(stderr, "%d\n", i);
      fprintf(stderr, "%d\n", *p);
      fprintf(stderr, "%d\n",  i * (*p));
*/
/*
      int size = 10;
      float a[size];
      float b[size];
      float c[size];
      for (unsigned i = 0; i < size; ++i) {
            a[i] = b[i] = 1.0f;
            c[i] = 0;
      }

      vectorAdd(a, b, c, size);
      for (unsigned j = 0; j < size; ++j) {
            fprintf(stderr, "%f\n", c[j]);
      }

      if (checkVectors(a, b, size)) {
            fprintf(stderr, "%s\n", "Equals vectors");
      } else {
            fprintf(stderr, "%s\n", "Differents vectors");
      }

      if (checkVectors(a, c, size)) {
            fprintf(stderr, "%s\n", "Equals vectors");
      } else {
            fprintf(stderr, "%s\n", "Differents vectors");
      }
*/

/*
      int m[SIZE][SIZE];
      int r[SIZE][SIZE];
      int v[SIZE][SIZE];

      initMatrix(m, 3, SIZE, SIZE);
      //
      initMatrix(r, 3, SIZE, SIZE);
      initMatrix(v, 0, SIZE, SIZE);
      addMatrixs(r , m, v, SIZE, SIZE);

      for (unsigned i = 0; i < SIZE; ++i) {
            for (unsigned j = 0; j < SIZE; ++j) {
                  fprintf(stderr, "%d\n", v[i][j]);
            }
      }
*/
/*
      float* f = NULL;
      float* g = NULL;
      float* h = NULL;
      mi_malloc_float(&f, SIZE * SIZE);
      mi_malloc_float(&g, SIZE * SIZE);
      mi_malloc_float(&h, SIZE * SIZE);

      if (f == NULL || g == NULL || h == NULL) {
            fprintf(stderr, "Error allocating memory... %s\n", strerror(errno));
            exit(1);
      }

      for (;;) {
            unsigned long last = get_time();
            initVector(f, 1, SIZE * SIZE);
            initVector(g, 2, SIZE * SIZE);
            addVector(f, g, h, SIZE * SIZE);
            unsigned long now = get_time();
            fprintf(stderr, "\nVector time %lf ms\n", (now - last) / 1000000.0);

            last = get_time();
            initMatrix_float(f, 1.0, SIZE, SIZE);
            initMatrix_float(g, 1.0, SIZE, SIZE);
            addMatrix(f, g, h, SIZE, SIZE);

            now = get_time();
            fprintf(stderr, "\nMatrix time %lf ms\n", (now - last) / 1000000.0);
      }
      free(g); free(h); free(f);
      fprintf(stdout, "Time now nano sec : %ld\n", get_time());
*/

      char* f = NULL;
      char* g = NULL;
      char* h = NULL;
      mi_malloc_char(&f, SIZE * SIZE);
      mi_malloc_char(&g, SIZE * SIZE);
      mi_malloc_char(&h, SIZE * SIZE);
      initVector_char(f, 1, SIZE * SIZE);
      initVector_char(g, 2, SIZE * SIZE);

      float* t = NULL;
      mi_malloc_float(&t, SIZE * SIZE);
      initVector(t, 2, SIZE * SIZE);

      for (;;) {
            unsigned long last = get_time();
            fprintf(stderr, "%lf\n", sum(t, SIZE * SIZE));
            unsigned long now = get_time();
            fprintf(stderr, "\nVector time %lf ms\n", (now - last) / 1000000.0);
      }
      exit(0);
}
