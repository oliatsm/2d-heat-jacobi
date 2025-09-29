
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>   // <-- Required for memset
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#define OFFSET(x, y, m) (((x)*(m)) + (y))

int file_output(double *A, int n, int m, const char *file_name){

    FILE* output;
    output = fopen(file_name,"w");

    if(output == NULL){
        printf("File not created.\n");
        return 1;
    }
    else{

        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                fprintf(output,"%.4f, ",A[OFFSET(i,j,m)]);
            }
            fprintf(output,"\n");
        }
    }

    fclose(output);
    return 0;
}

__global__
void initialize_cu(double * A, double * Anew, int m, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){
    
        A[OFFSET(0,i,m)] = 1.0;
        Anew[OFFSET(0,i,m)] = 1.0;
    }
}

void initialize(double *h_A, double *h_Anew, double *d_A, double *d_Anew, int m, int n)
{
    const unsigned int bytes = sizeof(double)*n*m;

    memset(h_A, 0, bytes);
    memset(h_Anew, 0, bytes);
 
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, h_Anew, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize; 

    initialize_cu<<<numBlocks, blockSize>>>(d_A, d_Anew, m, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    
}

__global__
void stencil_cu(double * A, double * Anew, int m, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x+1;
    int j = blockIdx.y*blockDim.y + threadIdx.y+1;

    if(i<(n-1) && j<(m-1)){
    
        Anew[OFFSET(j, i, m)] = 0.25 * ( A[OFFSET(j, i+1, m)] + A[OFFSET(j, i-1, m)] + A[OFFSET(j-1, i, m)] + A[OFFSET(j+1, i, m)]);
    }
}

__global__ void reduce0(double *A, double *Anew, int n, int m, double *d_max){
    extern __shared__ double sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i <n*m )
        sdata[tid] = fabs(A[i]-Anew[i]);
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        if (tid % (2 * s) == 0) {
            // sdata[tid] += sdata[tid + s];
            if(sdata[tid] < sdata[tid + s])
                sdata[tid] = sdata[tid + s];   
        }
        __syncthreads();
    }
    if (tid == 0){
        d_max[blockIdx.x] = sdata[0];
    }
}



double calcNext(double *h_A, double *h_Anew,double *d_A, double *d_Anew, int m, int n)
{

    dim3 block(16,16);
    dim3 grid((m + block.x - 1) / block.x,(n + block.y - 1) / block.y);


    stencil_cu<<<grid, block>>>(d_A, d_Anew, m, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    double error = 0.0;

    int blockSize = 256;
    int numBlocks = (n*m + blockSize - 1) / blockSize;

    size_t maxBytes = numBlocks * sizeof(double);

    double *h_max = (double*)malloc(maxBytes);
    double *d_max;
    cudaMalloc((void**)&d_max, maxBytes);
    
    reduce0<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_A,d_Anew,n,m,d_max);
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_max, d_max, maxBytes, cudaMemcpyDeviceToHost);

    error = h_max[0];
    for (int i = 1; i < numBlocks; ++i) {
        if(error < h_max[i])
            error = h_max[i];
    }
    
    return error;
}

__global__
void swap_cu(double *A, double *Anew, int m, int n){

    int i = blockIdx.x*blockDim.x + threadIdx.x+1;
    int j = blockIdx.y*blockDim.y + threadIdx.y+1;

    if(i<(n-1) && j<(m-1)){
        A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];        
    }

}
        
void swap(double *d_A, double *d_Anew, int m, int n)
{

    dim3 block(16,16);
    dim3 grid((m + block.x - 1) / block.x,(n + block.y - 1) / block.y);


    swap_cu<<<grid, block>>>(d_A, d_Anew, m, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

}

void deallocate(double *A, double *Anew, double *d_A, double *d_Anew)
{
    free(A);
    free(Anew);
    cudaFree(d_A);
    cudaFree(d_Anew);
}

