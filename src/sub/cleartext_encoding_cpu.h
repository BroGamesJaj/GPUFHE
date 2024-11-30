#ifndef CLEARTEXT_ENCODING_H
#define CLEARTEXT_ENCODING_H

#include <vector>
#include <inttypes.h>

namespace cleartext_encoding
{
    // Utility functions
    int mod(int x, int mod);
    int mod_pow(int base, int exp, int modulo);
    void bitreversal(std::vector<int>& poly);
    int find_primitive_root(int mod);
    std::vector<int> get_prime_factors(int n);

    // Encoding and Decoding functions
    std::vector<int> encode(const std::vector<int> &input, int n, int64_t MOD, int factor = 1);
    std::vector<int> decode(const std::vector<int> &poly, int n, int64_t MOD, int factor = 1);

    // Bit field encoding
    void bit_field_encode(std::vector<int> &vec, int factor, int64_t MOD);

    // NTT and INTT functions
    void NTT(std::vector<int> &poly, const std::vector<int> &roots, int n, int64_t MOD);
    void INTT(std::vector<int> &poly, const std::vector<int> &roots, int n, int64_t MOD);

    // Precomputation functions
    std::vector<int> precomputeRoots(int n, int64_t MOD, bool inverse = false);
    std::vector<int> find_divisors(int n);


    // Test function
    void ClearTextEncodingTest();
}

#endif