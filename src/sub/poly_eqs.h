#pragma once


#include <iostream>
#include <cuda_runtime.h>
#include "poly.h"
namespace poly_eqs {
    
    Polinomial PolyMult_cpu(const Polinomial& p1, const Polinomial& p2);

    Polinomial PolyMult_cpu(const Polinomial& p1, int64_t c);

    __global__ void PolyMult_gpu(int64_t *poly_1, int64_t *poly_2, int64_t *result, size_t poly_size);

    Polinomial PolyAdd_cpu(const Polinomial& p1, const Polinomial& p2);

    __global__ void PolyAdd_gpu(int64_t* poly_1, int64_t* poly_2, int64_t* result, size_t n);
    
    Polinomial PolySub_cpu(const Polinomial& p1, const Polinomial& p2);

    __global__ void PolySub_gpu(int64_t *poly_1, int64_t *poly_2, int64_t *result, size_t n);

    std::pair<Polinomial, Polinomial> PolyDiv_cpu(Polinomial& dividend, Polinomial& divisor);

    /*depricated 
    std::pair<Polinomial, Polinomial> PolyDiv_gpu(Polinomial& dividend, Polinomial& divisor);

    void PolyDivW_gpu(int64_t* remainder_d, int64_t* quotient, int64_t *divisor_d, size_t dividendSize, size_t divisorSize);*/

    __global__ void PolyDiv_gpu(int64_t* remainder_d, int64_t* quotient_d, int64_t *divisor_d, size_t dividendSize, size_t divisorSize);

    __global__ void PolyMultSub_gpu(size_t n, size_t i, int64_t coeff_div, int64_t *multiplier, int64_t *result);

    __global__ void ModCenter_gpu(int64_t* poly, int64_t modulo, size_t n);

    void ReducePoly_gpu(int64_t* poly, int64_t* modulus, int64_t modulo, size_t n);

    void ReducePoly_gpu(int64_t* poly, int64_t* modulus, int64_t modulo, size_t originalN, size_t n);
}