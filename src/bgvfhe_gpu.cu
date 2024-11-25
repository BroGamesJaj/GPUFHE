#include <stdio.h>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <random>
#include <inttypes.h>

__global__ void add(int* a, int* b, int* c){
    int i = threadIdx.x + blockIdx.y * blockDim.x;
    c[i] = a[i] + b[i];
}

__global__ void PolyMult_gpu(int* a, int* b, int* c, int size){
    int i = threadIdx.x + blockIdx.y * blockDim.x;
    if(i < size){
        c[i]  = a[i] * b[i];
    }
}
void init_poly(uint64_t *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (uint64_t)rand() / RAND_MAX;
    }
}
int main(){

    Polinomial array(10);
    Polinomial array2(10);
    Polinomial array3(19);
    std::random_device rd;                     // Seed for randomness
    std::mt19937 gen(rd());                    // Mersenne Twister generator
    std::uniform_int_distribution<size_t> dis(1, 10); // Uniform distribution [1, 10]

    init_poly(array.getCoeffPointer(), array.getSize()); 
    init_poly(array2.getCoeffPointer(), array2.getSize()); 
    array3 = poly_eqs::PolyMult(array,array2);
    for (int i=0; i<array3.getSize(); i++) 
    { 
       printf( "%" PRIu64, array3[i]); 
       if (i != 0) 
        printf("x^%d",i) ; 
       if (i != array3.getSize()-1) 
       printf(" + "); 
    } 
}