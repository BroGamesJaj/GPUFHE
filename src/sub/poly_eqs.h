#pragma once

#include "poly.h"
#include <iostream>
#include <cuda_runtime.h>

namespace poly_eqs {
    
    Polinomial PolyMult_cpu(Polinomial p1, Polinomial p2);

    __global__ void PolyMult_gpu(uint64_t *poly_1, uint64_t *poly_2, uint64_t *result, size_t poly_size);

    Polinomial PolyAdd_cpu(Polinomial p1, Polinomial p2);

    __global__ void PolyAdd_gpu(uint64_t *poly_1, uint64_t *poly_2, uint64_t *result);
    
    Polinomial PolySub_cpu(Polinomial p1, Polinomial p2);

    __global__ void PolySub_gpu(uint64_t *poly_1, uint64_t *poly_2, uint64_t *result);

    std::pair<Polinomial, Polinomial> PolyDiv_cpu( Polinomial dividend,  Polinomial divisor);

    __global__ void PolyDiv_gpu(uint64_t* dividend,  uint64_t* divisor, uint64_t* quotient, uint64_t* remainder, size_t degree);

    
}