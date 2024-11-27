#pragma once

#include "poly.h"
#include <iostream>

namespace poly_eqs {
    
    Polinomial PolyMult_cpu(Polinomial p1, Polinomial p2);

    __global__ void PolyMult_gpu(uint64_t *poly_1, uint64_t *poly_2, uint64_t *result, size_t poly_1_size, size_t poly_2_size);

    Polinomial PolyAdd_cpu(Polinomial p1, Polinomial p2);

    __global__ void PolyAdd_gpu(uint64_t *poly_1, uint64_t *poly_2, uint64_t *result, size_t poly_1_size, size_t poly_2_size);
}