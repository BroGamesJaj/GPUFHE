#include <stdio.h>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <random>
#include <inttypes.h>
#include <cstdint>

__global__ void add(int* a, int* b, int* c, int N){
    int i = threadIdx.x + blockIdx.y * blockDim.x;
    if (i < N) c[i] = a[i] + b[i];
}

__managed__ int vector_a[256], vector_b[256], vector_c[256];

int main(){

    Polinomial array(10);
    Polinomial array2(10);
    Polinomial array3(10);
    std::random_device rd;                     // Seed for randomness
    std::mt19937 gen(rd());                    // Mersenne Twister generator
    std::uniform_int_distribution<size_t> dis(1, 10); // Uniform distribution [1, 10]

    // Fill the array with random numbers
    for (size_t i = 0; i < array.getSize(); ++i) {
        array[i] = dis(gen); // Generate random number and assign to array
    }
    for (size_t i = 0; i < array2.getSize(); ++i) {
        array2[i] = dis(gen); // Generate random number and assign to array
    }

    int size = 10*sizeof(int);
    int* h_A = (float*)malloc(size);
    int* h_B = (float*)malloc(size);
    int* h_C = (float*)malloc(size);
    int *d_a;
    int *d_b;
    int *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =
            (10 + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, 10);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int e : c) printf("%d, ", e);
    printf("\n");

    for (int i=0; i<array3.getSize(); i++) 
    { 
       printf( "%" PRIu64, array3[i]); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != array3.getSize()-1) 
       printf(" + "); 
    } 
}