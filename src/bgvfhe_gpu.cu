#include <stdio.h>
#include "sub/poly.h"
#include "sub/poly_eqs.h"
#include <random>
#include <inttypes.h>

__global__ void add(int* a, int* b, int* c){
    int i = threadIdx.x + blockIdx.y * blockDim.x;
    c[i] = a[i] + b[i];
}

__managed__ int vector_a[256], vector_b[256], vector_c[256];

int main(){

    Polinomial array(10);
    Polinomial array2(10);
    Polinomial array3(19);
    std::random_device rd;                     // Seed for randomness
    std::mt19937 gen(rd());                    // Mersenne Twister generator
    std::uniform_int_distribution<size_t> dis(1, 10); // Uniform distribution [1, 10]

    // Fill the array with random numbers
    for (size_t i = 0; i < array.getSize(); ++i) {
        array[i] = dis(gen); // Generate random number and assign to array
    }
    for (size_t i = 0; i < array2.getSize(); ++i) {
        array2[i] = dis(gen); // Generate random number and assign to array
    }
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