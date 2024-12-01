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

void DivTest() {
    printf("Test for Polynomial substration\n");
    Polinomial dividend(N);
    Polinomial divisor(N);


    init_poly(dividend.getCoeffPointer(), dividend.getSize());
    init_poly(divisor.getCoeffPointer(), divisor.getSize());

    std::pair<Polinomial, Polinomial> res = poly_eqs::PolyDiv_cpu(dividend, divisor);

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        res = poly_eqs::PolyDiv_cpu(dividend, divisor);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    std::pair<Polinomial, Polinomial> res_gpu = poly_eqs::PolyDiv_gpu(dividend, divisor);

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        res_gpu = poly_eqs::PolyDiv_gpu(dividend, divisor);
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("Benchmarking Device Pointer GPU implementation...\n");
    int64_t *remainder_d, *divisor_d;
    int64_t* quotient = (int64_t*)malloc(sizeof(int64_t) * dividend.getSize() - divisor.getSize() + 1);
    cudaMalloc(&remainder_d, sizeof(int64_t) * dividend.getSize());
    cudaMalloc(&divisor_d, sizeof(int64_t) * divisor.getSize());
    cudaMemcpy(remainder_d, dividend.getCoeffPointer(), sizeof(int64_t) * dividend.getSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(divisor_d, divisor.getCoeffPointer(), sizeof(int64_t) * divisor.getSize(), cudaMemcpyHostToDevice);
    poly_eqs::PolyDiv_gpu(remainder_d, quotient, divisor_d, dividend.getSize(), divisor.getSize());
    double gpu2_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(remainder_d, dividend.getCoeffPointer(), sizeof(int64_t) * dividend.getSize(), cudaMemcpyHostToDevice);
        cudaMemcpy(divisor_d, divisor.getCoeffPointer(), sizeof(int64_t) * divisor.getSize(), cudaMemcpyHostToDevice);
        double start_time = get_time();
        poly_eqs::PolyDiv_gpu(remainder_d, quotient, divisor_d, dividend.getSize(), divisor.getSize());
        double end_time = get_time();
        gpu2_total_time += end_time - start_time;
    }
    double gpu2_avg_time = gpu2_total_time / 20.0;

    bool correct = true;
    for (int i = 0; i < res_gpu.first.getSize(); i++) {
        if(res_gpu.first[i] - res.first[i] != 0){
            correct = false;
            break;
        }
    }

    bool correct2 = true;
    for (int i = 0; i < dividend.getSize() - divisor.getSize() + 1; i++) {
        if(quotient[i] - res.first[i] != 0){
            correct2 = false;
            break;
        }
    }
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("GPU Device Pointer average time: %f milliseconds\n", gpu2_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu2_avg_time);
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    printf("Results2 are %s\n", correct2 ? "correct" : "incorrect");
}

Polinomial GeneratePrivateKey(int64_t coeff_modulus, GeneralArray<int64_t> poly_modulus){
    if(coeff_modulus != 0 && poly_modulus.getSize() != 0){
        Polinomial randomPoly = poly::randomTernaryPoly(coeff_modulus, poly_modulus);
    
        printf("private key sk:\n");
        randomPoly.print();
        return randomPoly;
    }else{
        throw std::runtime_error("coefficient or poly_modulus is not set");
    }
}

void PublicKeyTest(Polinomial pk0, Polinomial sk, Polinomial a, Polinomial e, int64_t plaintext_modulus){
    Polinomial temp1 = poly_eqs::PolyMult_cpu(a,sk);
    //printf("temp 1 after mult a * sk\n");
    //temp1.print();
    Polinomial temp2 = poly_eqs::PolyMult_cpu(e,plaintext_modulus);
    //printf("temp 2 after mult e * plaintext_modulus\n");
    //temp2.print();
    Polinomial reconstructed_pk0 = poly_eqs::PolyAdd_cpu(temp1,temp2);
    //printf("b after adding temp1 + temp2\n");
    //reconstructed_pk0.print();
    if(pk0 == reconstructed_pk0){
        printf("public key 0s are same\n");
    }else{
        printf("public key 0s are NOT same\n");
        pk0.print();
        reconstructed_pk0.print();
    }

    
}

std::pair<Polinomial,Polinomial> GeneratePublicKey(Polinomial& sk, int64_t coeff_modulus, GeneralArray<int64_t>& poly_modulus, int64_t plaintext_modulus){
    Polinomial e = poly::randomNormalPoly(coeff_modulus,poly_modulus);
    Polinomial a = poly::randomUniformPoly(coeff_modulus,poly_modulus);
    //printf("calc start\n");
    //printf("e\n");
    //e.print();
    //printf("a\n");
    //a.print();
    //printf("sk\n");
    //sk.print();
    Polinomial temp1 = poly_eqs::PolyMult_cpu(a,sk);
    //printf("temp 1 after mult a * sk\n");
    //temp1.print();

    Polinomial temp2 = poly_eqs::PolyMult_cpu(e,plaintext_modulus);
    //printf("temp 2 after mult e * plaintext_modulus\n");
    //temp2.print();

    Polinomial b = poly_eqs::PolyAdd_cpu(temp1,temp2);
    //printf("b after adding temp1 + temp2\n");
    //b.print();

    //PublicKeyTest(b,sk,a,e,plaintext_modulus);
    return std::make_pair(b,a);
}



bool isSmallNorm(const Polinomial& poly, int64_t bound) {
    for (int64_t coef : poly.getCoeff()) { // Iterate over coefficients of the polynomial
        if (std::abs(coef) > bound) {
            return false; // Coefficient exceeds the allowed bound
        }
    }
    return true; // All coefficients are within the bound
}

std::pair<Polinomial, Polinomial> asymetricEncryption(Polinomial pk0, Polinomial pk1, Polinomial msg, int64_t plaintext_modulus, int64_t coef_modulus, GeneralArray<int64_t> poly_modulus){
    Polinomial u = poly::randomTernaryPoly(coef_modulus,poly_modulus);
    Polinomial e0 = poly::randomNormalPoly(coef_modulus,poly_modulus);
    Polinomial e1 = poly::randomNormalPoly(coef_modulus,poly_modulus);

    Polinomial c0 = poly_eqs::PolyAdd_cpu(poly_eqs::PolyAdd_cpu(poly_eqs::PolyMult_cpu(pk0,u),e0),msg);
    Polinomial c1 = poly_eqs::PolyAdd_cpu(poly_eqs::PolyMult_cpu(pk1,u),e1);
    //printf("c0\n");
    //c0.print();
    //c1.print();
    return std::make_pair(c0,c1);
}

Polinomial decrypt(Polinomial c0, Polinomial c1, Polinomial sk, int64_t plaintext_modulus){
    Polinomial sk_c1 = poly_eqs::PolyMult_cpu(c1,sk);
    Polinomial msg = poly_eqs::PolySub_cpu(c0,sk_c1);
    msg.polyMod(plaintext_modulus);
    msg.print();
    return msg;
}

double computeNoiseNorm(const Polinomial& poly) {
    double norm = 0.0;
    for (int i = 0; i < poly.getSize(); ++i) {
        norm += poly[i] * poly[i];  // Sum of squares of coefficients
    }
    return sqrt(norm);  // L2 norm (Euclidean norm)
}

bool isNoiseSmallEnough(const Polinomial& noise, double threshold) {
    double norm = computeNoiseNorm(noise);
    return norm < threshold;  // Check if the noise norm is below the threshold
}

int main(){
    printf("started\n");
    int64_t n = 4;

    int64_t coef_modulus = 1024;
    for (size_t i = 0; i < 5; i++) {}
    GeneralArray<int64_t> poly_modulus = poly::initPolyModulus(n);
    printf("poly_modulus:\n");
    poly_modulus.print();
    int64_t plaintext_modulus = 7;
    Polinomial sk = GeneratePrivateKey(coef_modulus, poly_modulus);
    auto result = GeneratePublicKey(sk, coef_modulus, poly_modulus, plaintext_modulus);
    printf("PK generator ended\n");
    //printf("%d",result);
    Polinomial pk0 = result.first;
    Polinomial pk1 = result.second;
    Polinomial msg = poly::randomUniformPoly(coef_modulus, poly_modulus, plaintext_modulus);
    printf("MSG:\n");
    msg.print();
    auto encryption_result = asymetricEncryption(pk0,pk1,msg,plaintext_modulus,coef_modulus,poly_modulus);

    Polinomial decrypted_msg = decrypt(encryption_result.first, encryption_result.second,sk,plaintext_modulus);
    printf("decrypted MSG:\n");
    decrypted_msg.print();
    if (decrypted_msg == msg) {
        printf("Decryption successful\n");
    } else {
        printf("Decryption failed\n");
    }
    if(isNoiseSmallEnough(encryption_result.first,512)){
        printf("Good noise\n");
    } else {
        printf("Bad noise\n");
    }

}