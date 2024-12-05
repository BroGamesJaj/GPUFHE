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
#include "tests.cuh"
#include "sub/cypertext_eqs.h"


namespace bgvfhe_gpu {
    Polinomial GeneratePrivateKey(int64_t coeff_modulus, GeneralArray<int64_t> poly_modulus);

    std::pair<Polinomial,Polinomial> GeneratePublicKey(Polinomial& sk, int64_t coeff_modulus, GeneralArray<int64_t>& poly_modulus, int64_t plaintext_modulus);
    std::pair<Polinomial, Polinomial> asymetricEncryption(Polinomial pk0, Polinomial pk1, Polinomial msg, int64_t plaintext_modulus, int64_t coef_modulus, GeneralArray<int64_t> poly_modulus, int64_t degree);
    
    Polinomial decrypt(Polinomial c0, Polinomial c1, Polinomial sk, int64_t plaintext_modulus);
    Polinomial decrypt_quad(Polinomial c0, Polinomial c1, Polinomial c2, Polinomial sk, int64_t plaintext_modulus);


    int64_t logBase(int64_t value, int base);

    GeneralArray<int64_t> int2Base(int value, int base, int& digitCount);

    GeneralArray<Polinomial*> poly2Base(Polinomial poly, int base);

    std::pair<Polinomial,Polinomial> Relinearization(Polinomial c0, Polinomial c1, Polinomial c2, GeneralArray<std::pair<Polinomial,Polinomial>*> eks, int base, int64_t coef_modulus, int64_t poly_modulus);
    
    bool isSmallNorm(const Polinomial& poly, int64_t bound);

    bool isNoiseSmallEnough(const Polinomial& noise, double threshold);

}