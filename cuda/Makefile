CC = nvcc

CFLAGS = -Xcompiler -fopenmp 

all: laplace2d.cu jacobi.cu
	$(CC) laplace2d.cu jacobi.cu -o laplace $(CFLAGS)

clean:
	rm -f laplace

