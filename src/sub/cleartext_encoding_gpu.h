#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <utility>

namespace cleartext_encoding {
    extern int* roots_d;
    extern int* roots_inverse_d;

    __device__ int mod(int x, int64_t mod);
    __device__ int mod_pow(int base, int exp, int64_t MOD);
    __device__ int reverseBits(int n, int numBits);

    __global__ void find_primitive_root(int mod, int* factors_d, int factorSize, int64_t phi, int* primitive_root_d);
    __global__ void get_prime_factors(bool* primes_d, int* factors, int primesSize, int n);
    __global__ void sieve(bool* primes_d, int n);
    __global__ void bitreversal(int* poly_d, int* polyOut_d, int n);
    __global__ void NTT(int* poly_d, int* roots_d, int64_t MOD, int n);
    __global__ void INTT(int* poly_d, int64_t MOD, int invN);
    __global__ void bit_field_encode(int* input_d, int n, int64_t MOD, int factor);
    __global__ void EncodingTestKernel(int a, int MOD);

    int find_primitive_root_wrapper(int mod);
    std::pair<int*, int> get_prime_factors_wrapper(int n);
    int* precomputeRoots_wrapper(int n, int64_t MOD, int* factors_d, int factorSize, bool inverse = false);
   void precomputeRoots(int primitive_root, int *roots, int64_t MOD, int n, bool inverse = false);
    void precompute(int64_t MOD, int n);
    void encode(int* poly, int n, int64_t MOD, int factor);
    void decode(int* poly, int n, int64_t MOD, int factor);

    void EncodingTest();
}