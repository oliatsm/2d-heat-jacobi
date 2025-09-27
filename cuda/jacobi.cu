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
#include <omp.h>

#include "laplace2d.h"


int main(int argc, char** argv)
{
    const int n = 4096/4;
    const int m = 4096/4;
    const int iter_max = 1000;
    
    const double tol = 1.0e-6;
    double error = 1.0;

    const unsigned int bytes = sizeof(double)*n*m;

    double *h_A    = (double*)malloc(bytes);
    double *h_Anew = (double*)malloc(bytes);

    double *d_A, *d_Anew;

    cudaMalloc((void**)&d_A,bytes);
    cudaMalloc((void**)&d_Anew,bytes);

    initialize(h_A, h_Anew, d_A, d_Anew, m, n);

    printf("A: %f %f %f, Anew %f %f %f \n",h_A[0],h_A[n],h_A[n*m-1],h_Anew[0],h_Anew[n],h_Anew[n*m-1]);
        
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    double st = omp_get_wtime();
    int iter = 0;
   
    while ( error > tol && iter < iter_max )
    {
        error = calcNext(h_A, h_Anew, d_A, d_Anew, m, n);
        swap(d_A, d_Anew, m, n);
        
        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;

    }

    double runtime = omp_get_wtime() - st;
 
    printf(" total: %f s\n", runtime);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, h_Anew, bytes, cudaMemcpyHostToDevice);

    file_output(h_Anew,n,m,"output-swap.txt");

    deallocate(h_A, h_Anew,d_A,d_Anew);

    return 0;
}
