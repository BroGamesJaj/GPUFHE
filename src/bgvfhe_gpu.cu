#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <random>
#include <inttypes.h>
#include <cuda_runtime.h>
#define N 10000
#define M 1

__global__ void add(int* a, int* b, int* c){
    int i = threadIdx.x + blockIdx.y * blockDim.x;
    c[i] = a[i] + b[i];
}

__global__ void PolyMult_gpu(uint64_t* poly_1, uint64_t* poly_2, uint64_t* result, size_t poly_1_size, size_t poly_2_size){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= poly_1_size + poly_2_size - 1) return;
    uint64_t sum = 0;
    for (int j = 0; j < poly_1_size; j++) {
        if (i - j >= 0 && i - j < poly_2_size) {
            sum += poly_1[j] * poly_2[i - j];
        }
    }
    result[i] = sum;
}
void init_poly(uint64_t *array, int n) {
    std::random_device rd;                     // Seed for randomness
    std::mt19937 gen(rd());                    // Mersenne Twister generator
    std::uniform_int_distribution<size_t> dis(1, 10);
    for (size_t i = 0; i < n; ++i) {
        array[i] = dis(gen); // Generate random number and assign to array
    }
}

double get_time() {
    static LARGE_INTEGER frequency;
    static BOOL initialized = FALSE;
    
    // Initialize the frequency only once
    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = TRUE;
    }

    // Get the current counter value
    LARGE_INTEGER current_time;
    QueryPerformanceCounter(&current_time);

    // Calculate the time in seconds
    return (double)current_time.QuadPart / frequency.QuadPart;
}

int main(){
    size_t size1 = N * sizeof(uint64_t);
    size_t size2 = M * sizeof(uint64_t);
    size_t size_out = (M + N - 1) * sizeof(uint64_t);
    Polinomial array(N);
    Polinomial array2(M);
    Polinomial array3((M + N -1));
    uint64_t *d_a, *d_b, *d_c;

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
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        array3 = poly_eqs::PolyMult(array,array2);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;
    
    
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
    cudaMalloc(&d_a, size1);
    cudaMalloc(&d_b, size2);
    cudaMalloc(&d_c, size_out);
    cudaMemset(d_c, 0, size_out);
    cudaMemcpy(d_a, array.getCoeffPointer(), size1, cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, array2.getCoeffPointer(), size2, cudaMemcpyHostToDevice );

    int block_num = (M * N + 256 - 1) / 256;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        PolyMult_gpu<<<block_num,256>>>(d_a, d_b, d_c, array.getSize(), array.getSize());
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    cudaMemcpy(array3.getCoeffPointer(), d_c, size_out, cudaMemcpyDeviceToHost);

    for (int i=0; i<array3.getSize(); i++) 
    { 
       printf( "%" PRIu64, array3[i]); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != array3.getSize()-1) 
       printf(" + "); 
    } 

    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}