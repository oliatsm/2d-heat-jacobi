
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>   // <-- Required for memset
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#define OFFSET(x, y, m) (((x)*(m)) + (y))

__global__
void initialize_cu(double * A, double * Anew, int m, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){
    
        A[i] = 1.0;
        Anew[i] = 1.0;
    }
}

void initialize(double *h_A, double *h_Anew, double *d_A, double *d_Anew, int m, int n)
{
    const unsigned int bytes = sizeof(double)*n*m;

    memset(h_A, 0, bytes);
    memset(h_Anew, 0, bytes);

    cudaMalloc((void**)&d_A,bytes);
    cudaMalloc((void**)&d_Anew,bytes);

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
    
    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Anew, d_Anew, bytes, cudaMemcpyDeviceToHost);
    
}


double calcNext(double *h_A, double *h_Anew,double *d_A, double *d_Anew, int m, int n)
{
    double error = 0.0;

    // dim3 block(16,16);
    // dim3 grid((m + block.x - 1) / block.x,
    //           (n + block.y - 1) / block.y);

    // calcNext_cu<<<grid, block>>>(d_A, d_Anew, m, n);
    // cudaDeviceSynchronize();
    //  printf("A: %f %f %f, Anew %f %f %f \n",h_A[0],h_A[n],h_A[n*m-1],h_Anew[0],h_Anew[n],h_Anew[n*m-1]);
    for( int j = 1; j < n-1; j++)
    {
        for( int i = 1; i < m-1; i++ )
        {
            h_Anew[OFFSET(j, i, m)] = 0.25 * ( h_A[OFFSET(j, i+1, m)] + h_A[OFFSET(j, i-1, m)]
                                           + h_A[OFFSET(j-1, i, m)] + h_A[OFFSET(j+1, i, m)]);
            error = fmax( error, fabs(h_Anew[OFFSET(j, i, m)] - h_A[OFFSET(j, i , m)]));
        }
    }
    return error;
}
        
void swap(double *A, double *Anew, int m, int n)
{
    for( int j = 1; j < n-1; j++)
    {
        for( int i = 1; i < m-1; i++ )
        {
            A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];    
        }
    }
}

void deallocate(double *A, double *Anew, double *d_A, double *d_Anew)
{
    free(A);
    free(Anew);
    cudaFree(d_A);
    cudaFree(d_Anew);
}

int file_output(double *A, int n, int m, char* file_name){

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