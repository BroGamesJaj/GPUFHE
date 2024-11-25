#include <stdio.h>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <random>
#include <inttypes.h>
#define N 2000

__global__ void add(int* a, int* b, int* c){
    int i = threadIdx.x + blockIdx.y * blockDim.x;
    c[i] = a[i] + b[i];
}

__global__ void PolyMult_gpu(uint64_t* a, uint64_t* b, uint64_t* c, int size){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < 2 * size - 1) {
        c[i] = 0; // Initialize result coefficient
        for (int j = 0; j < size; j++) {
            if (i - j >= 0 && i - j < size) {
                c[i] += a[j] * b[i - j];
            }
        }
    }
}
void init_poly(uint64_t *array, int n) {
    std::random_device rd;                     // Seed for randomness
    std::mt19937 gen(rd());                    // Mersenne Twister generator
    std::uniform_int_distribution<size_t> dis(1, 10);
    for (size_t i = 0; i < n; ++i) {
        array[i] = dis(gen); // Generate random number and assign to array
    }
}
int main(){
    size_t size = N * sizeof(uint64_t);
    size_t size_out = (2 * N - 1) * sizeof(uint64_t);
    Polinomial array(size);
    Polinomial array2(size);
    Polinomial array3(size_out);
    uint64_t *d_a, *d_b, *d_c;
    std::random_device rd;                     // Seed for randomness
    std::mt19937 gen(rd());                    // Mersenne Twister generator
    std::uniform_int_distribution<size_t> dis(1, 10); // Uniform distribution [1, 10]

    init_poly(array.getCoeffPointer(), array.getSize()); 
    init_poly(array2.getCoeffPointer(), array2.getSize()); 
    for (int i=0; i<array.getSize(); i++) 
    { 
       printf( "%" PRIu64, array[i]); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != array.getSize()-1) 
       printf(" + "); 
    } 
    printf("\n");
    for (int i=0; i<array2.getSize(); i++) 
    { 
       printf( "%" PRIu64, array2[i]); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != array2.getSize()-1) 
       printf(" + "); 
    } 
    printf("\n");
    printf("\n");
    array3 = poly_eqs::PolyMult(array,array2);
    for (int i=0; i<array3.getSize(); i++) 
    { 
       printf( "%" PRIu64, array3[i]); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != array3.getSize()-1) 
       printf(" + "); 
    } 
    printf("\n");
    printf("\n");
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size_out);
    cudaMemset(d_c, 0, size_out);
    cudaMemcpy(d_a, array.getCoeffPointer(), size, cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, array2.getCoeffPointer(), size, cudaMemcpyHostToDevice );

    int block_num = (N + 256 - 1) / 256;
    PolyMult_gpu<<<block_num,256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(array3.getCoeffPointer(), d_c, size_out, cudaMemcpyDeviceToHost);

    for (int i=0; i<array3.getSize(); i++) 
    { 
       printf( "%" PRIu64, array3[i]); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != array3.getSize()-1) 
       printf(" + "); 
    } 
}