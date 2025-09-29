
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>   

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

// ──────────────────────────────────────────────────────────────
// Coordinate convention used in this code:
//
//  - X dimension  →  columns   (i index)      →  max size = m
//  - Y dimension  →  rows      (j index)      →  max size = n
//
//  - OFFSET(row, col, m) gives the 1-D index for a 2-D array
//    stored in row-major order, where `m` = total number of columns.
//    Example: element at (row=j, col=i) is A[OFFSET(j, i, m)].
//
//  In CUDA kernels:
//      threadIdx.x / blockIdx.x  → column index
//      threadIdx.y / blockIdx.y  → row index
// ──────────────────────────────────────────────────────────────

#define OFFSET(x, y, m) (((x)*(m)) + (y))

// funtion for exporting the 2D Arrays on textfiles
int file_output(double *A, int m, int n, const char *file_name){

    FILE* output;
    output = fopen(file_name,"w");

    if(output == NULL){
        printf("File not created.\n");
        return 1;
    }
    else{

        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                fprintf(output,"%.4f, ",A[OFFSET(j,i,m)]);
            }
            fprintf(output,"\n");
        }
    }

    fclose(output);
    return 0;
}

// kernel to initialize first row of 2-d plates to 1
__global__
void initialize_cu(double * d_A, double * d_Anew, int m, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<m){
    
        d_A[OFFSET(0,i,m)] = 1.0;
        d_Anew[OFFSET(0,i,m)] = 1.0;
    }
}

void initialize(double *h_A, double *h_Anew, double *d_A, double *d_Anew, int m, int n)
{
    int bytes = sizeof(double)*n*m;

    //Initialise arrays on host and device

    memset(h_A, 0, bytes);
    memset(h_Anew, 0, bytes);
 
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, h_Anew, bytes, cudaMemcpyHostToDevice);

    // Launch 1D grid covering only the first row (n columns).
    // We need exactly n threads (one per column), so we compute the
    // number of blocks to cover n elements with blockSize threads each.

    int blockSize = 256;
    int numBlocks = (m + blockSize - 1) / blockSize; 

    initialize_cu<<<numBlocks, blockSize>>>(d_A, d_Anew, m, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    
}

// Stencil acces pattern
// Anew is calculated as the average of the neighnor cells.
__global__
void stencil_cu(double * d_A, double * d_Anew, int m, int n)
{   
    // global indicies for rows (j) and columns (i)
    // i range from 1 to m-2, j ragne from 1 to n-2
    // i=0, j=0, i=m-1 and j=n-1 contain boundary conditions, they don't change

    int i = blockIdx.x*blockDim.x + threadIdx.x+1; // columns
    int j = blockIdx.y*blockDim.y + threadIdx.y+1; //rows

    if(i<(m-1) && j<(n-1)){
    
        d_Anew[OFFSET(j, i, m)] = 0.25 * ( d_A[OFFSET(j, i+1, m)] + d_A[OFFSET(j, i-1, m)] 
                                       + d_A[OFFSET(j-1, i, m)] + d_A[OFFSET(j+1, i, m)]);
    }
}

// Reduction for error calculation
__global__ 
void max_reduce(double *A, double *Anew, int m, int n, double *d_max){

    //shared memory: size blockDim (blockSize) 
    //1d access, no spatial dependence of data
    extern __shared__ double sdata[]; 

    // tid: shared memory index
    // i: global memory index
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i <n*m )
        sdata[tid] = fabs(A[i]-Anew[i]);
    __syncthreads();

    // Calculate max for each block
    for(int s = 1; s < blockDim.x; s *= 2){
        if (tid % (2 * s) == 0) {
            if(sdata[tid] < sdata[tid + s])
                sdata[tid] = sdata[tid + s];   
        }
        __syncthreads();
    }

    // each block returns local max value
    // d_max size: number of blocks (numBlocks)
    // d_max index range from 0 to numBlocks
    if (tid == 0){
        d_max[blockIdx.x] = sdata[0];
    }
}



double calcNext(double *d_A, double *d_Anew, int m, int n, double *h_max, double *d_max, int maxSize)
{
    // Separate Anew calculation and error calculation

    //*****************************
    //   Anew : stencil kernel    
    //****************************
    cudaError_t cuda_err;

    dim3 block(16,16);
    dim3 grid((m + block.x - 1) / block.x,(n + block.y - 1) / block.y);


    stencil_cu<<<grid, block>>>(d_A, d_Anew, m, n);

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(cuda_err));
    }

    cudaDeviceSynchronize();

    //*****************************
    //   error: max reduce    
    //****************************
    double error = 0.0;

    // 1d me
    int blockSize = maxSize;
    int numBlocks = (n*m + blockSize - 1) / blockSize;
    
    max_reduce<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_A,d_Anew,m,n,d_max);
    cudaDeviceSynchronize();
    
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(cuda_err));
    }

    cudaMemcpy(h_max, d_max, maxSize*sizeof(double), cudaMemcpyDeviceToHost);

    // final reduction for max
    error = h_max[0];
    for (int i = 1; i < numBlocks; ++i) {
        if(error < h_max[i])
            error = h_max[i];
    }
    

    return error;
}

// Copy Anew to A
__global__
void swap_cu(double *d_A, double *d_Anew, int m, int n){

    int i = blockIdx.x*blockDim.x + threadIdx.x+1;
    int j = blockIdx.y*blockDim.y + threadIdx.y+1;

    if(i<(m-1) && j<(n-1)){
        d_A[OFFSET(j, i, m)] = d_Anew[OFFSET(j, i, m)];        
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

void deallocate(double *h_A, double *h_Anew, double *d_A, double *d_Anew)
{
    free(h_A);
    free(h_Anew);
    cudaFree(d_A);
    cudaFree(d_Anew);
}

