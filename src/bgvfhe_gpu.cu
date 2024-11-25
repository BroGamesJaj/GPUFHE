#include <stdio.h>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <inttypes.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 10000000

__global__ void poly_mult(int* a, int* b, int* c, int size){
    int i = blockIdx.y * blockIdx.x + threadIdx.x;
    if(i < size){
        c[i] = a[i] * b[i];
    }
}

void init_array(int *array, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = (int)rand() / RAND_MAX;
    }
}

int main(){
    int *h_array1, *h_array2, *h_array3;
    int *d_array1, *d_array2, *d_array3;
    size_t size = N * sizeof(int);
    size_t size3 = size+size-1;
    h_array1 = (int*)malloc(size);
    h_array2 = (int*)malloc(size);
    h_array3 = (int*)malloc(size3);
    init_array(h_array1,size);
    init_array(h_array2,size);

    cudaMalloc(&d_array1,size);
    cudaMalloc(&d_array2,size);
    cudaMalloc(&d_array3,size3);

    cudaMemcpy(d_array1, h_array1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array2, size, cudaMemcpyHostToDevice);

    int block_num = (N + 256 -1) / 256;
    poly_mult<<<block_num,256>>>(d_array1, d_array2, d_array3, size);

    cudaDeviceSynchronize();

    cudaMemcpy(h_array3, d_array3, size3, cudaMemcpyDeviceToHost);

    for (int i=0; i<size3; i++) 
    { 
       printf( "%" PRIu64, *h_array3); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != size3-1) 
       printf(" + "); 
    } 
}