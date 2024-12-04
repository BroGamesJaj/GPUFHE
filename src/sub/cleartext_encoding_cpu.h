#pragma once

#include <vector>
#include <inttypes.h>
#include <utility>

namespace cleartext_encoding_cpu
{
    int mod(int x, int mod);
    int mod_pow(int base, int exp, int modulo);
    void bitreversal(std::vector<int>& poly);
    int find_primitive_root(int mod);
    std::pair<int*, int> get_prime_factors(int n);
    std::vector<int> get_prime_factors_vector(int n);

    std::vector<int> encode(const std::vector<int> &input, int n, int64_t MOD, int factor = 1);
    std::vector<int> decode(const std::vector<int> &poly, int n, int64_t MOD, int factor = 1);

    void bit_field_encode(std::vector<int> &vec, int factor, int64_t MOD);

    void NTT(std::vector<int> &poly, const std::vector<int> &roots, int n, int64_t MOD);
    void INTT(std::vector<int> &poly, const std::vector<int> &roots, int n, int64_t MOD);

    std::vector<int> precomputeRoots(int n, int64_t MOD, bool inverse = false);


    // Test function
    void ClearTextEncodingTest(std::vector<int> roots, std::vector<int> rootsInverse);
}