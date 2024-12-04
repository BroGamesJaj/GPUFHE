#include "cleartext_encoding_gpu.h"
#include <iostream>
#include <utility>
#include <cmath>
#include <algorithm>

#include <vector>
#include "cleartext_encoding_cpu.h"

namespace cleartext_encoding {
    int *roots_d = nullptr;
    int *roots_inverse_d = nullptr;

    __device__ int mod(int x, int64_t mod) {
        return ((x % mod) + mod) % mod;
    }

    __device__ int mod_pow(int base, int exp, int MOD) {
        int result = 1;
        base = mod(base, MOD);
        while (exp > 0) {
            if (exp % 2 == 1)
                result = mod(result * base, MOD);
            base = mod(base * base, MOD);
            exp /= 2;
        }
        return result;
    }

    __global__ void find_primitive_root(int mod, int* factors_d, int factorSize, int64_t phi, int* primitive_root_d) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x + 2;
        if (idx > phi) return;
        bool is_primitive = true;
        for (int i = 0; i < factorSize; i++) 
        {
            if (factors_d[i] != 0 && mod_pow(idx, phi / factors_d[i], mod) == 1)
            {
                is_primitive = false;
                break;
            }
        }
        if (is_primitive)
            (*primitive_root_d) = idx;
    }

    int find_primitive_root_wrapper(int mod) {
        int phi = mod - 1;
        auto [factors_d, factorSize] = get_prime_factors_wrapper(phi);
        int *primitive_root_d;
        int primitive_root = -1;
        cudaMalloc(&primitive_root_d, sizeof(int));
        cudaMemset(primitive_root_d, -1, sizeof(int));
        int blockSize = 256;
        int numBlocks = (mod + blockSize - 4) / blockSize; //mod - 3 threads
        find_primitive_root<<<numBlocks, blockSize>>>(mod, factors_d, factorSize, phi, primitive_root_d);
        cudaDeviceSynchronize();
        cudaMemcpy(&primitive_root, primitive_root_d, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(factors_d);
        cudaFree(primitive_root_d);
        return primitive_root;
    }

    __global__ void get_prime_factors(bool* primes_d, int* factors, int primesSize, int n) {
        __shared__ int count;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx == 0) count = 0;
        __syncthreads();
        if (idx >= primesSize) return;
        if (!primes_d[idx]) return;
        idx += 2;
        while (n % idx == 0) {
            int i = atomicAdd(&count, 1);
            factors[i] = idx;
            n /= idx;
        }
    }

    __global__ void sieve(bool* primes_d, int n) { // n = floor(sqrt(inp))
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx = idx * 2 + 1;
        if (idx == 1) idx = 2;
        if (idx > floorf(sqrtf(n))) return;
        int index = idx;

        while (idx * index <= n) {
            primes_d[idx * index - 2] = false;
            index++;
        }
    }

    std::pair<int*, int> get_prime_factors_wrapper(int n) {
        bool* primes_d;
        int sqrN2 = floor(sqrt(n/2));
        int primesSize = n/2-1;
        cudaMalloc(&primes_d, sizeof(bool) * primesSize);
        cudaMemset(primes_d, true, sizeof(bool) * primesSize);
        int blockSize = 256;
        int numBlocks = (sqrN2 + blockSize - 1) / blockSize;
        sieve<<<numBlocks, blockSize>>>(primes_d, primesSize + 1);
        cudaDeviceSynchronize();

        int factorSize = floor(log2(n));
        int *factors_d;
        cudaMalloc(&factors_d, sizeof(int) * (factorSize));

        int numBlocks2 = (primesSize + blockSize - 1) / blockSize;
        get_prime_factors<<<numBlocks2, blockSize>>>(primes_d, factors_d, primesSize, n);
        cudaDeviceSynchronize();

        cudaFree(primes_d);
        return {factors_d, factorSize};
    }

    __global__ void bitreversal(int *poly_d, int *polyOut_d, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        int bits = log2f(n);
        polyOut_d[reverseBits(idx, bits)] = poly_d[idx];
    }

    __device__ int reverseBits(int n, int numBits) {
        int reversed = 0;
        for (int i = 0; i < numBits; ++i) {
            reversed <<= 1;
            reversed |= (n & 1);
            n >>= 1;
        }
        return reversed;
    }

    int* precomputeRoots_wrapper(int n, int64_t MOD, int* factors_d, int factorSize, bool inverse) {
        int blockSize = 256;
        
        int numBlocksPrimitive = (MOD + blockSize - 4) / blockSize; //mod - 3 threads
        int *primitive_root_d;
        cudaMalloc(&primitive_root_d, sizeof(int));
        find_primitive_root<<<numBlocksPrimitive, blockSize>>>(MOD, factors_d, factorSize, MOD - 1, primitive_root_d);
        cudaDeviceSynchronize();

        int *roots = (int*)malloc(n * sizeof(int));
        int primitive_root;
        cudaMemcpy(&primitive_root, primitive_root_d, sizeof(int), cudaMemcpyDeviceToHost);
        precomputeRoots(primitive_root, roots, MOD, n, inverse);
        cudaDeviceSynchronize();
        cudaFree(primitive_root_d);
        return roots_d;
    }

    __global__ void NTT(int *poly_d, int *roots_d, int64_t MOD, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        //int64_t len = powf(2, idx + 1);
        //if (len > n) return;
        if (idx != 0) return;
        for (int len = 2; len <= n; len <<= 1) {
        int step = n / len;
        for (int i = 0; i < n; i += len)
        {
            for (int j = 0; j < len / 2; ++j)
            {
                int u = poly_d[i + j];
                int v = mod(poly_d[i + j + len / 2] * roots_d[j * step], MOD);
                poly_d[i + j] = mod((u + v), MOD);
                poly_d[i + j + len / 2] = mod((u - v + MOD), MOD);
            }
        }
        }
    }

    __global__ void INTT(int *poly_d, int n,  int64_t MOD, int invN) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        poly_d[idx] = mod(1LL * poly_d[idx] * invN, MOD);
    }

    __global__ void bit_field_encode(int *input_d, int n, int64_t MOD, int factor) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        input_d[idx] = mod(input_d[idx] * factor, MOD);
    }

    void precomputeRoots(int primitive_root, int *roots, int64_t MOD, int n, bool inverse) {
        int root = cleartext_encoding_cpu::mod_pow(primitive_root, (MOD - 1) / n, MOD);
        if (inverse)
            root = cleartext_encoding_cpu::mod_pow(root, MOD - 2, MOD);
        roots[0] = 1;
        for (int i = 1; i < n; ++i)
            roots[i] = cleartext_encoding_cpu::mod(roots[i - 1] * root, MOD);
    }


    void precompute(int64_t MOD, int n) {
        int blockSize = 256;
        int numBlocksPrimitive = (MOD + blockSize - 4) / blockSize; //mod - 3 threads

        auto [factors_d, factorSize] = get_prime_factors_wrapper(MOD - 1);

        int *primitive_root_d;
        cudaMalloc(&primitive_root_d, sizeof(int));
        find_primitive_root<<<numBlocksPrimitive, blockSize>>>(MOD, factors_d, factorSize, MOD - 1, primitive_root_d);
        cudaDeviceSynchronize();

        int primitive_root;
        cudaMemcpy(&primitive_root, primitive_root_d, sizeof(int), cudaMemcpyDeviceToHost);

        int numBlocksRoots = (n + blockSize - 2) / blockSize;
        cudaMalloc(&roots_d, sizeof(int) * n);
        cudaMalloc(&roots_inverse_d, sizeof(int) * n);
        int *roots = (int*)malloc(n * sizeof(int));
        int *roots_inverse = (int*)malloc(n * sizeof(int));
        precomputeRoots(primitive_root, roots, MOD, n, false);
        precomputeRoots(primitive_root, roots_inverse, MOD, n, true);
        cudaMemcpy(roots_d, roots, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(roots_inverse_d, roots_inverse, n * sizeof(int), cudaMemcpyHostToDevice);
    }

    void encode(int *poly, int n, int64_t MOD, int factor) {
        int blockSize = 256;
        int *poly_d, *polyOut_d;
        cudaMalloc(&poly_d, sizeof(int) * n);
        cudaMemcpy(poly_d, poly, sizeof(int) * n, cudaMemcpyHostToDevice);

        int numBlocksBit = (n + blockSize - 2) / blockSize;
        bit_field_encode<<<numBlocksBit, blockSize>>>(poly_d, n, MOD, factor);
        
        cudaMalloc(&polyOut_d, sizeof(int) * n);
        cudaMemcpy(polyOut_d, poly_d, sizeof(int) * n, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        bitreversal<<<numBlocksBit, blockSize>>>(poly_d, polyOut_d, n);
        int numBlockNTT = (log2(n) + blockSize - 2) / blockSize;
        cudaDeviceSynchronize();
        NTT<<<numBlockNTT, blockSize>>>(polyOut_d, roots_d, MOD, n);
        cudaDeviceSynchronize();
        cudaMemcpy(poly, polyOut_d, sizeof(int) * n, cudaMemcpyDeviceToHost);
    }

    void decode(int *poly, int n, int64_t MOD, int factor) {
        int blockSize = 256;
        int *poly_d, *polyOut_d;
        cudaMalloc(&poly_d, sizeof(int) * n);
        cudaMemcpy(poly_d, poly, sizeof(int) * n, cudaMemcpyHostToDevice);

        int numBlocksBit = (n + blockSize - 2) / blockSize;
        
        cudaMalloc(&polyOut_d, sizeof(int) * n);
        cudaMemcpy(polyOut_d, poly_d, sizeof(int) * n, cudaMemcpyDeviceToDevice);

        bitreversal<<<numBlocksBit, blockSize>>>(poly_d, polyOut_d, n);
        int numBlockNTT = (log2(n) + blockSize - 2) / blockSize;
        cudaDeviceSynchronize();
        NTT<<<numBlockNTT, blockSize>>>(polyOut_d, roots_inverse_d, MOD, n);
        cudaDeviceSynchronize();
        INTT<<<numBlocksBit, blockSize>>>(polyOut_d, n, MOD, cleartext_encoding_cpu::mod_pow(n, MOD - 2, MOD));
        cudaDeviceSynchronize();
        bit_field_encode<<<numBlocksBit, blockSize>>>(polyOut_d, n, MOD, cleartext_encoding_cpu::mod_pow(factor, MOD - 2, MOD));
        cudaDeviceSynchronize();
        cudaMemcpy(poly, polyOut_d, sizeof(int) * n, cudaMemcpyDeviceToHost);
    }


    void EncodingTest() {
        int n = 16;
        int factor = 1024;
        int64_t MOD = 12289;
        int *poly = (int*)malloc(n * sizeof(int));
        poly[0] = 1;
        poly[1] = 2;
        poly[2] = 3;
        poly[3] = 4;
        poly[4] = 5;
        poly[5] = 6;
        poly[6] = 7;
        poly[7] = 8;
        poly[8] = 9;
        poly[9] = 10;
        poly[10] = 11;
        poly[11] = 12;
        poly[12] = 13;
        poly[13] = 14;
        poly[14] = 15;
        poly[15] = 16;
        precompute(MOD, n);
        std::vector<int> poly_cpu(n);
        for (int i = 0; i < n; i++) poly_cpu[i] = poly[i];
        std::vector<int> encodedCpu = cleartext_encoding_cpu::encode(poly_cpu, n, MOD, factor);
        
        encode(poly, n, MOD, factor);
        printf("encoded: ");
        for (int i = 0; i < n; i++) printf("%d ", poly[i]);
        printf("\n");
        ///
        for (int i = 0; i < n; i++) poly_cpu[i] = poly[i];
        std::vector<int> decodedCpu = cleartext_encoding_cpu::decode(poly_cpu, n, MOD, factor);

        decode(poly, n, MOD, factor);

        printf("decoded: ");
        for (int i = 0; i < n; i++) printf("%d ", poly[i]);
        printf("\n");

        
        printf("cpu encoded: ");
        for (int i = 0; i < n; i++) printf("%d ", encodedCpu[i]);
        printf("\n");

        printf("cpu decoded: ");
        for (int i = 0; i < n; i++) printf("%d ", decodedCpu[i]);
        printf("\n");

        for (int i = 0; i < n; i++) poly[i] = encodedCpu[i];

        decode(poly, n, MOD, factor);

        printf("gpu decoded from cpu encode: ");
        for (int i = 0; i < n; i++) printf("%d ", poly[i]);
        printf("\n");
    }

    __global__ void EncodingTestKernel(int a, int MOD) {
        printf("GPUMOD: %d\n", mod(a, MOD));
        printf("GPU modpow: %d\n", mod_pow(a, 2, MOD));
    }
}

/*
mod
mod_pow
find_primitive_root
get_prime_factors
bitreversal
precompute
sieve?
*/

/*
NTT
INTT
bit_field_encode
encode
decode
*/