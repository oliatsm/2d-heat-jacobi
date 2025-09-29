CC = nvcc
CFLAGS = #-Xcompiler "-Wall -Werror" #-fopenmp 

all: laplace2d.cu jacobi.cu
	$(CC) laplace2d.cu jacobi.cu -o laplace $(CFLAGS)

clean:
	rm -f laplace

