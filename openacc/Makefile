CC = nvc

NVFLAGS = -acc -Minfo=all -Wall -Wextra 

CFLAGS = -lm

all: laplace2d.c jacobi.c laplace2d.h
	$(CC) $(NVFLAGS) laplace2d.c jacobi.c -o laplace $(CFLAGS)

clean:
	rm -f laplace

