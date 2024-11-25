#include <stdio.h>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <inttypes.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 10000000

__global__ void polyMult_gpu(int* a, int* b, int* c, int size){
    int i = blockIdx.y * blockIdx.x + threadIdx.x;
    if(i < size){
        c[i] = a[i] * b[i];
    }
}

void PolyMult_cpu(int* p1, int* p2, int* p3, int size){

        for (int i=0; i<size; i++) { 
            for (int j=0; j<size; j++){
                p3[i+j] = p1[i]*p2[j]; 
            }
        } 
    }
void init_array(int *array, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = (int)rand() / RAND_MAX;
    }
}

int main(){
    int *h_array1, *h_array2, *h_array_gpu, *h_array_cpu;
    int *d_array1, *d_array2, *d_array_gpu;
    size_t size = N * sizeof(int);
    h_array1 = (int*)malloc(size);
    h_array2 = (int*)malloc(size);
    h_array_gpu = (int*)malloc(size);
    h_array_cpu = (int*)malloc(size);
    init_array(h_array1,size);
    init_array(h_array2,size);
    /*
    cudaMalloc(&d_array1,size);
    cudaMalloc(&d_array2,size);
    cudaMalloc(&d_array_gpu,size);

    cudaMemcpy(d_array1, h_array1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array2, size, cudaMemcpyHostToDevice);

    int block_num = (N + 256 -1) / 256;
    polyMult_gpu<<<block_num,256>>>(d_array1, d_array2, d_array_gpu, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_array_gpu, d_array_gpu, size, cudaMemcpyDeviceToHost);
    
    for (int i=0; i<size; i++) 
    { 
       printf( "%d" PRIu64, *h_array_gpu); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != size-1) 
       printf(" + "); 
    } 
    printf("\n");
    */
    PolyMult_cpu(h_array1, h_array2, h_array_cpu, size);
    for (int i=0; i<size; i++) 
    { 
       printf( "%d" PRIu64, *h_array_cpu); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != size-1) 
       printf(" + "); 
    } 
}