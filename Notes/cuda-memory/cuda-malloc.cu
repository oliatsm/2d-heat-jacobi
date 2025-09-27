#include <stdio.h>
#include <stdlib.h>

void memory_allocations(int *h_a, int *d_a,const int size){

    h_a = malloc(size);

}

void free_memory(int *h_a, int *d_a, const int size){

    free(h_a);
    cudaFree(d_a);

}

int main(){
    int *h_A, *d_A;
    int N = 100;

    memory_allocations(h_A,d_A,N);

    // calculate_arrays();
    // print();

    free_memory(h_A,d_A,N);
    return 0;
}