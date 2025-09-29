CC = gcc

CFLAGS = -O3 -fopenmp -lm

all: laplace2d.c jacobi.c laplace2d.h
	$(CC) laplace2d.c jacobi.c -o laplace $(CFLAGS)

clean:
	rm -f laplace

