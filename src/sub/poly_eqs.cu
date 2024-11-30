#include "poly_eqs.h"
#include "assert.h"
#include <chrono>

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

    Polinomial PolyMult_cpu( Polinomial p1, int64_t c){
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

    std::pair<Polinomial, Polinomial> PolyDiv_cpu(Polinomial& dividend, Polinomial& divisor) {
        while (divisor[divisor.getSize() - 1] == 0) {
            divisor.getCoeff().pop_back();
        }
        if (divisor.getSize() < 2) {
            throw new std::runtime_error("divisor must be at least degree 1");
        }

        size_t dividendSize = dividend.getSize();
        size_t divisorSize = divisor.getSize();
        Polinomial quotient(dividendSize - divisorSize + 1);
        Polinomial remainder(dividendSize);

        remainder = dividend;

        for (int i = dividendSize - 1; i >= divisorSize - 1; --i) {
            int coeff_div = remainder[i] / divisor[divisorSize - 1];
            quotient[i - divisorSize + 1] = coeff_div;

            for (int j = 0; j < divisorSize; ++j) {
                remainder[i - j] -= coeff_div * divisor[divisorSize - 1 - j];
            }
        }
        return {quotient, remainder};
    }

    /*depricated 
    std::pair<Polinomial, Polinomial> PolyDiv_gpu(Polinomial& dividend, Polinomial& divisor) {
        while (divisor[divisor.getSize() - 1] == 0) {
            divisor.getCoeff().pop_back();
        }
        if (divisor.getSize() < 2) {
            throw new std::runtime_error("divisor must be at least degree 1");
        }

        size_t dividendSize = dividend.getSize();
        size_t divisorSize = divisor.getSize();

        Polinomial quotient = Polinomial(dividendSize - divisorSize + 1);
        Polinomial remainder = Polinomial(dividend);

        int64_t *remainder_d;
        int64_t *divisor_d;

        cudaMalloc(&remainder_d, dividendSize * sizeof(int64_t));
        cudaMalloc(&divisor_d, divisorSize * sizeof(int64_t));

        cudaMemcpy(remainder_d, remainder.getCoeffPointer(), dividendSize * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(divisor_d, divisor.getCoeffPointer(), divisorSize * sizeof(int64_t), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (divisorSize + blockSize - 2) / blockSize;

        for (int i = dividendSize - 1; i >= divisorSize - 1; --i) {
            int coeff_div = remainder[i] / divisor[divisorSize - 1];
            quotient[i - divisorSize + 1] = coeff_div;

            PolyMultSub_gpu<<<numBlocks, blockSize>>>(divisorSize, i, coeff_div, divisor_d, remainder_d);
            cudaDeviceSynchronize();
            if (i >= divisorSize)
                cudaMemcpy(remainder.getCoeffPointer() + (i - 1), remainder_d + (i - 1), sizeof(int64_t), cudaMemcpyDeviceToHost);
        }

        cudaMemcpy(remainder.getCoeffPointer(), remainder_d, dividendSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaFree(remainder_d);
        cudaFree(divisor_d);

        return {quotient, remainder};
    }

    void PolyDivW_gpu(int64_t* remainder_d, int64_t* quotient, int64_t *divisor_d, size_t dividendSize, size_t divisorSize) {
        int blockSize = 256;
        int numBlocks = (divisorSize + blockSize - 2) / blockSize;

        int64_t remainder_host;
        int64_t divisor_host;

        cudaMemcpy(&remainder_host, remainder_d + (dividendSize - 1), sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&divisor_host, divisor_d + (divisorSize - 1), sizeof(int64_t), cudaMemcpyDeviceToHost);

        for (int i = dividendSize - 1; i >= divisorSize - 1; --i) {
            int coeff_div = remainder_host / divisor_host;
            quotient[i - divisorSize + 1] = coeff_div;

            PolyMultSub_gpu<<<numBlocks, blockSize>>>(divisorSize, i, coeff_div, divisor_d, remainder_d);
            cudaDeviceSynchronize();
            if (i >= divisorSize) 
                cudaMemcpy(&remainder_host, remainder_d + (i - 1), sizeof(int64_t), cudaMemcpyDeviceToHost);
        }
    }*/

    __global__ void PolyDiv_gpu(int64_t* remainder_d, int64_t* quotient_d, int64_t *divisor_d, size_t dividendSize, size_t divisorSize) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= divisorSize) return;

        for (int i = dividendSize - 1; i >= divisorSize - 1; --i) {
            __syncthreads();
            int coeff_div = remainder_d[i] / divisor_d[divisorSize - 1];
            if (idx == 0) quotient_d[i - divisorSize + 1] = coeff_div;

            remainder_d[i - idx] -= coeff_div * divisor_d[divisorSize - 1 - idx];
        }
    }

    __global__ void PolyMultSub_gpu(size_t n, size_t i, int64_t coeff_div, int64_t *multiplier, int64_t *result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;;
        if (idx >= n) return;
        result[i - idx] -= coeff_div * multiplier[n - 1 - idx];
    }
}
