#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <random>
#include <inttypes.h>
#include <cuda_runtime.h>
#define N 1000
#define M 1000  

void init_poly(uint64_t *array, int n) {
    std::random_device rd;                     // Seed for randomness
    std::mt19937 gen(rd());                    // Mersenne Twister generator
    std::uniform_int_distribution<size_t> dis(1, 10);
    for (size_t i = 0; i < n; ++i) {
        array[i] = dis(gen); // Generate random number and assign to array
    }
}

double get_time() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}
int main(){
    size_t size1 = N * sizeof(uint64_t);
    size_t size2 = M * sizeof(uint64_t);
    size_t size_out = M * sizeof(uint64_t);
    Polinomial array(N);
    Polinomial array2(M);
    Polinomial array3(M);
    Polinomial array_gpu(M);
    uint64_t *d_a, *d_b, *d_c;

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        array3 = poly_eqs::PolyAdd_cpu(array,array2);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;
    
    printf("\n");
    cudaMalloc(&d_a, size1);
    cudaMalloc(&d_b, size2);
    cudaMalloc(&d_c, size_out);
    cudaMemset(d_c, 0, size_out);
    cudaMemcpy(d_a, array.getCoeffPointer(), size1, cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, array2.getCoeffPointer(), size2, cudaMemcpyHostToDevice );

    int block_num = (M + 256 - 1) / 256;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        poly_eqs::PolyAdd_gpu<<<block_num,256>>>(d_a, d_b, d_c, array.getSize(), array2.getSize());
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;
    cudaMemcpy(array_gpu.getCoeffPointer(), d_c, size_out, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < array_gpu.getSize(); i++) {
        if(array_gpu[i] - array3[i] != 0){
            correct = false;
            break;
        }
    }
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
/* 
int main(){
    size_t size1 = N * sizeof(uint64_t);
    size_t size2 = M * sizeof(uint64_t);
    size_t size_out = (M + N - 1) * sizeof(uint64_t);
    Polinomial array(N);
    Polinomial array2(M);
    Polinomial array3((M + N -1));
    Polinomial array_gpu((M + N -1));
    uint64_t *d_a, *d_b, *d_c;


    init_poly(array.getCoeffPointer(), array.getSize()); 
    init_poly(array2.getCoeffPointer(), array2.getSize()); 

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        array3 = poly_eqs::PolyMult_cpu(array,array2);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;
    
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
        poly_eqs::PolyMult_gpu<<<block_num,256>>>(d_a, d_b, d_c, array.getSize(), array2.getSize());
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;
    cudaMemcpy(array_gpu.getCoeffPointer(), d_c, size_out, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < array_gpu.getSize(); i++) {
        if(array_gpu[i] - array3[i] != 0){
            correct = false;
            break;
        }
    }
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
*/