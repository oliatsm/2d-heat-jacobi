/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "timer.h" //for GpuTimer
#include "laplace2d.h"


int main(int argc, char** argv)
{
    const int n = 4096/8;
    const int m = 4096/6;
    const int iter_max = 1000;
    
    const double tol = 1.0e-6;
    double error = 1.0;

    const unsigned int bytes = sizeof(double)*n*m;

    double *h_A    = (double*)malloc(bytes);
    double *h_Anew = (double*)malloc(bytes);

    double *d_A, *d_Anew;

    cudaMalloc((void**)&d_A,bytes);
    cudaMalloc((void**)&d_Anew,bytes);

    // For Error calculation on device
    int maxSize = 256;
    // max array is used for local max calculation in each block
    // max[numBlocks]
    int maxBytes = (n*m + maxSize - 1) / maxSize * sizeof(double);

    double *h_max = (double*)malloc(maxBytes);
    double *d_max;
    cudaMalloc((void**)&d_max, maxBytes);
    

    initialize(h_A, h_Anew, d_A, d_Anew, m, n);
       
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    GpuTimer timer;

    timer.Start();

    int iter = 0;
   
    while ( error > tol && iter < iter_max )
    {
        error = calcNext(d_A, d_Anew, m, n, h_max, d_max, maxSize);
        swap(d_A, d_Anew, m, n);
        
        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;

    }

    timer.Stop();

    printf(" total: %f s\n", timer.Elapsed()/1000.0);

    // Copy Data to host for output
    
    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Anew, d_Anew, bytes, cudaMemcpyDeviceToHost);

    file_output(h_Anew,n,m,"output.txt");

    deallocate(h_A, h_Anew,d_A,d_Anew);

    cudaFree(d_max);
    free(h_max);


    return 0;
}
