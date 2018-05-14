CC = g++
CFLAGS = -O2 -g -fopenmp -Wall -Wshadow -march=x86-64

PROGRAMS = main

all: $(PROGRAMS)

% : %.c
	$(CC) $(CFLAGS) $< -o $@ -lm

.PHONY : clean

clean:
	rm $(PROGRAMS)
