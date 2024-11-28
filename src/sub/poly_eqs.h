#pragma once

#include "poly.h"
#include <iostream>

namespace poly_eqs {
    
    Polinomial PolyMult_cpu(Polinomial p1, Polinomial p2);

    __global__ void PolyMult_gpu(int64_t *poly_1, int64_t *poly_2, int64_t *result, size_t poly_size);

    Polinomial PolyAdd_cpu(Polinomial p1, Polinomial p2);

    __global__ void PolyAdd_gpu(int64_t *poly_1, int64_t *poly_2, int64_t *result);
    
    Polinomial PolySub_cpu(Polinomial p1, Polinomial p2);

    __global__ void PolySub_gpu(int64_t *poly_1, int64_t *poly_2, int64_t *result);
}