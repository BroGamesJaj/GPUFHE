#include "poly_eqs.h"

namespace poly_eqs{
    Polinomial PolyMult_cpu( Polinomial p1, Polinomial p2){
        Polinomial prod(p1.getSize()+p2.getSize()-1);

        for (size_t i=0; i<p1.getSize(); i++) { 
            for (size_t j=0; j<p1.getSize(); j++){
                prod[i+j] += p1[i]*p2[j]; 
            }
        } 
        return prod;
    }

    Polinomial PolyMultConst_cpu( Polinomial p1, int c){
        Polinomial prod(p1.getSize());

        for (size_t i=0; i<p1.getSize(); i++) { 
            for (size_t j=0; j<p1.getSize(); j++){
                prod[i+j] += p1[i]*c; 
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

    std::pair<Polinomial, Polinomial> PolyDiv_cpu(Polinomial poly1, Polinomial poly2) {
        Polinomial quotient(1);
        Polinomial remainder = poly1;

        // Perform polynomial long division
        while (poly1.getSize() >= poly2.getSize()) {
            printf("poly1\n");
            poly1.print();
            int quotient_term = poly1.back() / poly2.back();  // Get the next quotient term

            // Create the product of poly2 and the current quotient term
            Polinomial product(poly1.getSize());
            
            product = PolyMultConst_cpu(poly2,quotient_term);
            printf("product\n");
            product.print();
            remainder = PolySub_cpu(poly1,product);
            
            // Remove leading zeros from the remainder
            while (remainder.getSize() > 0 && remainder.back() == 0) {
                remainder.pop_back();
            }

            poly1 = remainder;
        }

        quotient.getCoeff().resize(poly1.getSize() + poly2.getSize() - 1);
        quotient = poly1;
        remainder = poly1;

        return {quotient, remainder};
    }


    __global__ void PolyDiv_gpu(int64_t* dividend, int64_t* divisor, int64_t* quotient, int64_t* remainder, size_t degree) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure thread index is within bounds
    if (tid <= degree) {
        // Calculate the quotient coefficient for this degree
        int64_t coeff = dividend[tid] / divisor[tid];
        quotient[tid] = coeff;

        // Update the remainder for this degree
        remainder[tid] = dividend[tid] - coeff * divisor[tid];
    }
}
}

