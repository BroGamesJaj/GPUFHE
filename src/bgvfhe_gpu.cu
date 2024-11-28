#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <random>
#include <inttypes.h>
#include <cuda_runtime.h>
#define N 10

void init_poly(int64_t *array, int n) {
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

void AddTest(){
    printf("test for polynomial addition\n");
    size_t size1 = N * sizeof(int64_t);
    size_t size_out = N * sizeof(int64_t);
    Polinomial array(N);
    Polinomial array2(N);
    Polinomial array3(N);
    Polinomial array_gpu(N);
    int64_t *d_a, *d_b, *d_c;

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
    cudaMalloc(&d_b, size1);
    cudaMalloc(&d_c, size_out);
    cudaMemset(d_c, 0, size_out);
    cudaMemcpy(d_a, array.getCoeffPointer(), size1, cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, array2.getCoeffPointer(), size1, cudaMemcpyHostToDevice );

    int block_num = (N + 256 - 1) / 256;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        poly_eqs::PolyAdd_gpu<<<block_num,256>>>(d_a, d_b, d_c);
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
    printf("\n");
}

void SubTest(){
    printf("Test for Polynomial substration\n");
    size_t size1 = N * sizeof(int64_t);
    size_t size_out = N * sizeof(int64_t);
    Polinomial array(N);
    Polinomial array2(N);
    Polinomial array3(N);
    Polinomial array_gpu(N);
    int64_t *d_a, *d_b, *d_c;

    printf("Benchmarking CPU implementation...\n\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        array3 = poly_eqs::PolySub_cpu(array,array2);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    cudaMalloc(&d_a, size1);
    cudaMalloc(&d_b, size1);
    cudaMalloc(&d_c, size_out);
    cudaMemset(d_c, 0, size_out);
    cudaMemcpy(d_a, array.getCoeffPointer(), size1, cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, array2.getCoeffPointer(), size1, cudaMemcpyHostToDevice );

    int block_num = (N + 256 - 1) / 256;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        poly_eqs::PolySub_gpu<<<block_num,256>>>(d_a, d_b, d_c);
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
    printf("\n");
}
 
void MultTest(){
    printf("test for polynomial multiplication\n");
    size_t size1 = N * sizeof(int64_t);
    size_t size_out = (2 * N - 1) * sizeof(int64_t);
    Polinomial array(N);
    Polinomial array2(N);
    Polinomial array3((2 * N -1));
    Polinomial array_gpu((2 * N -1));
    int64_t *d_a, *d_b, *d_c;


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
    cudaMalloc(&d_b, size1);
    cudaMalloc(&d_c, size_out);
    cudaMemset(d_c, 0, size_out);
    cudaMemcpy(d_a, array.getCoeffPointer(), size1, cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, array2.getCoeffPointer(), size1, cudaMemcpyHostToDevice );

    int block_num = (2 * N + 256 - 1) / 256;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        poly_eqs::PolyMult_gpu<<<block_num,256>>>(d_a, d_b, d_c, array.getSize());
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
    printf("\n");
}

Polinomial GeneratePrivateKey(int64_t coeff_modulus, GeneralArray<int64_t> poly_modulus){
    if(coeff_modulus != 0 && poly_modulus.getSize() != 0){
        Polinomial randomPoly = poly::randomTernaryPoly(7, poly_modulus);
    
        printf("sk\n");
        randomPoly.print();
        return randomPoly;
    }else{
        throw std::runtime_error("coefficient or poly_modulus is not set");
    }
}

std::pair<Polinomial, Polinomial> GeneratePublicKey(Polinomial sk, int64_t coeff_modulus, GeneralArray<int64_t> poly_modulus, int64_t plaintext_modulus){
    Polinomial e = poly::randomNormalPoly(coeff_modulus,poly_modulus);
    Polinomial a = poly::randomUniformPoly(coeff_modulus,poly_modulus);

    Polinomial b = poly_eqs::PolyAdd_cpu(poly_eqs::PolyMult_cpu(a, sk),poly_eqs::PolyMult_cpu(e, plaintext_modulus));
    Polinomial side1 = poly_eqs::PolySub_cpu(b, poly_eqs::PolyMult_cpu(a,sk));
    printf("side1\n");
    side1.print();
    Polinomial side2 = poly_eqs::PolyMult_cpu(e,plaintext_modulus);
    printf("side2\n");
    side2.print();
    if(side1 == side2){
        printf("correct");
    }
    return {b, -a};
}

int main(){
    printf("started\n");
    int64_t n = 16;

    int64_t coef_modulus =874;
    GeneralArray<int64_t> poly_modulus = poly::initPolyModulus(n);
    int64_t plaintext_modulus = 7;
    Polinomial sk = GeneratePrivateKey(coef_modulus, poly_modulus);
    auto [pk0, pk1] = GeneratePublicKey(sk, coef_modulus, poly_modulus, plaintext_modulus);
    printf("pk0\n");
    pk0.print();
    printf("pk1\n");
    pk1.print();
}