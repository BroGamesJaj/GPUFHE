#include "poly_eqs.h"

namespace poly_eqs{
    Polinomial PolyMult_cpu(Polinomial p1, Polinomial p2){
        Polinomial prod(p1.getSize()+p2.getSize()-1);

        for (int i=0; i<p1.getSize(); i++) { 
            for (int j=0; j<p2.getSize(); j++){
                prod[i+j] += p1[i]*p2[j]; 
            }
        } 
        return prod;
    }

    __global__ void PolyMult_gpu(uint64_t* poly_1, uint64_t* poly_2, uint64_t* result, size_t poly_1_size, size_t poly_2_size){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= poly_1_size + poly_2_size - 1) return;
        uint64_t sum = 0;
        for (int j = 0; j < poly_1_size; j++) {
            if (i - j >= 0 && i - j < poly_2_size) {
                sum += poly_1[j] * poly_2[i - j];
            }
        }
        result[i] = sum;
    }

}