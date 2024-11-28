#include "poly_eqs.h"

namespace poly_eqs{
    Polinomial PolyMult_cpu(Polinomial p1, Polinomial p2){
        Polinomial prod(p1.getSize()+p2.getSize()-1);

        for (int i=0; i<p1.getSize(); i++) { 
            for (int j=0; j<p1.getSize(); j++){
                prod[i+j] += p1[i]*p2[j]; 
            }
        } 
        return prod;
    }

    __global__ void PolyMult_gpu(int64_t* poly_1, int64_t* poly_2, int64_t* result, size_t poly_size){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= 2 * poly_size - 1) return;
        int64_t sum = 0;
        for (int j = 0; j < poly_size; j++) {
            if (i - j >= 0 && i - j < poly_size) {
                sum += poly_1[j] * poly_2[i - j];
            }
        }
        result[i] = sum;
    }

    Polinomial PolyAdd_cpu(Polinomial p1, Polinomial p2){
        Polinomial prod(p1.getSize());

        for (size_t i = 0; i < p1.getSize(); i++) {
            prod[i] = p1[i] + p2[i];
        }
        return prod;
    }

    __global__ void PolyAdd_gpu(int64_t* poly_1, int64_t* poly_2, int64_t* result){
        int i = threadIdx.x + blockIdx.y * blockDim.x;
        result[i] = poly_1[i] + poly_2[i];
    }

    Polinomial PolySub_cpu(Polinomial p1, Polinomial p2){
        Polinomial prod(p1.getSize());

        for (size_t i = 0; i < p1.getSize(); i++) {
            prod[i] = p1[i] - p2[i];
        }
        return prod;
    }

    __global__ void PolySub_gpu(int64_t* poly_1, int64_t* poly_2, int64_t* result){
        int i = threadIdx.x + blockIdx.y * blockDim.x;
        result[i] = poly_1[i] - poly_2[i];
    }
}