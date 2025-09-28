
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h> // <-- Required for memset
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#define OFFSET(x, y, m) (((x) * (m)) + (y))
#define RAD 1

int file_output(double *A, int n, int m, const char *file_name)
{

    FILE *output;
    output = fopen(file_name, "w");

    if (output == NULL)
    {
        printf("File not created.\n");
        return 1;
    }
    else
    {

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                fprintf(output, "%.4f, ", A[OFFSET(i, j, m)]);
            }
            fprintf(output, "\n");
        }
    }

    fclose(output);
    return 0;
}

__global__ void initialize_cu(double *A, double *Anew, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {

        A[OFFSET(0, i, m)] = 1.0;
        Anew[OFFSET(0, i, m)] = 1.0;
    }
}

void initialize(double *h_A, double *h_Anew, double *d_A, double *d_Anew, int m, int n)
{
    const unsigned int bytes = sizeof(double) * n * m;

    memset(h_A, 0, bytes);
    memset(h_Anew, 0, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, h_Anew, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    initialize_cu<<<numBlocks, blockSize>>>(d_A, d_Anew, m, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

__global__ void stencil_cu(double *A, double *Anew, int m, int n)
{
    extern __shared__ double s_data[]; // size = (blockDim.x+2)*(blockDim.y+2)

    // global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x; // column index (x)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row index (y)

    // shared indices (offset by 1 for halo)
    int s_i = threadIdx.x + RAD; // shared column index
    int s_j = threadIdx.y + RAD; // shared row index

    int SDIM = blockDim.x + 2 * RAD; // number of columns in shared tile

    // initialize shared memory cell for this thread (or 0 if out of domain)
    if (i < m && j < n)
    {
        s_data[OFFSET(s_j, s_i, SDIM)] = A[OFFSET(j, i, m)];
    }
    else
    {
        s_data[OFFSET(s_j, s_i, SDIM)] = 0.0;
    }

    // load left & right halos (done by threads with threadIdx.x < RAD)
    if (threadIdx.x < RAD)
    {
        // left neighbor (global)
        if (i - RAD >= 0 && j < n)
        {
            s_data[OFFSET(s_j, s_i - RAD, SDIM)] = A[OFFSET(j, i - RAD, m)];
        }
        else
        {
            s_data[OFFSET(s_j, s_i - RAD, SDIM)] = 0.0;
        }

        // right neighbor (global): fetch element blockDim.x to the right
        if (i + blockDim.x < m && j < n)
        {
            s_data[OFFSET(s_j, s_i + blockDim.x, SDIM)] = A[OFFSET(j, i + blockDim.x, m)];
        }
        else
        {
            s_data[OFFSET(s_j, s_i + blockDim.x, SDIM)] = 0.0;
        }
    }

    // load top & bottom halos (done by threads with threadIdx.y < RAD)
    if (threadIdx.y < RAD)
    {
        // top neighbor
        int top_j = j - RAD;
        if (top_j >= 0 && i < m)
        {
            s_data[OFFSET(s_j - RAD, s_i, SDIM)] = A[OFFSET(top_j, i, m)];
        }
        else
        {
            s_data[OFFSET(s_j - RAD, s_i, SDIM)] = 0.0;
        }

        // bottom neighbor: fetch element blockDim.y down
        int bottom_j = j + blockDim.y;
        if (bottom_j < n && i < m)
        {
            s_data[OFFSET(s_j + blockDim.y, s_i, SDIM)] = A[OFFSET(bottom_j, i, m)];
        }
        else
        {
            s_data[OFFSET(s_j + blockDim.y, s_i, SDIM)] = 0.0;
        }
    }

    // load the four corner halos (to ensure all neighbor accesses are valid)
    // top-left
    if (threadIdx.x < RAD && threadIdx.y < RAD)
    {
        if (i - RAD >= 0 && j - RAD >= 0)
        {
            s_data[OFFSET(s_j - RAD, s_i - RAD, SDIM)] = A[OFFSET(j - RAD, i - RAD, m)];
        }
        else
        {
            s_data[OFFSET(s_j - RAD, s_i - RAD, SDIM)] = 0.0;
        }

        // top-right
        if (i + blockDim.x < m && j - RAD >= 0)
        {
            s_data[OFFSET(s_j - RAD, s_i + blockDim.x, SDIM)] = A[OFFSET(j - RAD, i + blockDim.x, m)];
        }
        else
        {
            s_data[OFFSET(s_j - RAD, s_i + blockDim.x, SDIM)] = 0.0;
        }

        // bottom-left
        if (i - RAD >= 0 && j + blockDim.y < n)
        {
            s_data[OFFSET(s_j + blockDim.y, s_i - RAD, SDIM)] = A[OFFSET(j + blockDim.y, i - RAD, m)];
        }
        else
        {
            s_data[OFFSET(s_j + blockDim.y, s_i - RAD, SDIM)] = 0.0;
        }

        // bottom-right
        if (i + blockDim.x < m && j + blockDim.y < n)
        {
            s_data[OFFSET(s_j + blockDim.y, s_i + blockDim.x, SDIM)] = A[OFFSET(j + blockDim.y, i + blockDim.x, m)];
        }
        else
        {
            s_data[OFFSET(s_j + blockDim.y, s_i + blockDim.x, SDIM)] = 0.0;
        }
    }

    __syncthreads();

    // only compute if the thread corresponds to a valid interior grid point
    // (we'll require neighbors i-1,i+1 and j-1,j+1 to exist)
    if (i >= 1 && j >= 1 && i < m - 1 && j < n - 1)
    {
        Anew[OFFSET(j, i, m)] = 0.25 * (s_data[OFFSET(s_j, s_i - 1, SDIM)] 
                                      + s_data[OFFSET(s_j, s_i + 1, SDIM)] 
                                      + s_data[OFFSET(s_j - 1, s_i, SDIM)] 
                                      + s_data[OFFSET(s_j + 1, s_i, SDIM)]);
    }
    // else: you may want to set boundary Anew[...] explicitly if needed
}

double calcNext(double *h_A, double *h_Anew, double *d_A, double *d_Anew, int m, int n)
{
    const unsigned int bytes = sizeof(double) * n * m;

    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    size_t smem = (block.x + 2 * RAD) * (block.y + 2 * RAD) * sizeof(double);

    stencil_cu<<<grid, block, smem>>>(d_A, d_Anew, m, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Post-kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Anew, d_Anew, bytes, cudaMemcpyDeviceToHost);

    double error = 0.0;

    for (int j = 1; j < n - 1; j++)
    {
        for (int i = 1; i < m - 1; i++)
        {
            error = fmax(error, fabs(h_Anew[OFFSET(j, i, m)] - h_A[OFFSET(j, i, m)]));
        }
    }

    return error;
}

__global__ void swap_cu(double *A, double *Anew, int m, int n)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < (n - 1) && j < (m - 1))
    {
        A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];
    }
}

void swap(double *d_A, double *d_Anew, int m, int n)
{

    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    swap_cu<<<grid, block>>>(d_A, d_Anew, m, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
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
