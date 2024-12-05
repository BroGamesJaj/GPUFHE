#pragma once
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <random>
#include <inttypes.h>
#include <cuda_runtime.h>
#include "sub/cleartext_encoding_cpu.h"
#include "bgvfhe_gpu.cuh"
namespace tests {
    
    double get_time();

    Polinomial GeneratePrivateKey(int64_t coeff_modulus, GeneralArray<int64_t> poly_modulus);


    std::pair<Polinomial,Polinomial> GeneratePublicKey(Polinomial& sk, int64_t coeff_modulus, GeneralArray<int64_t>& poly_modulus, int64_t plaintext_modulus);

    void AddTest();
    void SubTest();
    void MultTest();
    void DivTest();

    void PublicKeyTest(Polinomial pk0, Polinomial pk1, Polinomial sk, Polinomial a, Polinomial e, int64_t plaintext_modulus);
    void cMultTest(int64_t n, int64_t coef_modulus, int64_t plaintext_modulus, GeneralArray<int64_t> poly_modulus, Polinomial sk, std::pair<Polinomial,Polinomial> pk, int64_t batch);
    void cAddTest(int64_t n, int64_t coef_modulus, int64_t plaintext_modulus, GeneralArray<int64_t> poly_modulus, Polinomial sk, std::pair<Polinomial,Polinomial> pk, int64_t batch);
    void cSubTest(int64_t n, int64_t coef_modulus, int64_t plaintext_modulus, GeneralArray<int64_t> poly_modulus, Polinomial sk, std::pair<Polinomial,Polinomial> pk, int64_t batch);
}
