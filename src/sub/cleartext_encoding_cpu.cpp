#include "cleartext_encoding_cpu.h"

#include <iostream>

namespace cleartext_encoding_cpu
{
    int mod(int x, int mod)
    {
        return ((x % mod) + mod) % mod;
    }
    
    int mod_pow(int base, int exp, int modulo)
    {
        int result = 1;
        base = mod(base, modulo);
        while (exp > 0)
        {
            if (exp % 2 == 1)
                result = mod(result * base, modulo);
            base = mod(base * base, modulo);
            exp /= 2;
        }
        return result;
    }

    void NTT(std::vector<int> &poly, const std::vector<int> &roots, int n, int64_t MOD)
    {
        bitreversal(poly);
        for (int len = 2; len <= n; len <<= 1)
        {
            int step = n / len;
            for (int i = 0; i < n; i += len)
            {
                for (int j = 0; j < len / 2; ++j)
                {
                    int u = poly[i + j];
                    int v = mod(poly[i + j + len / 2] * roots[j * step], MOD);
                    poly[i + j] = mod((u + v), MOD);
                    poly[i + j + len / 2] = mod((u - v + MOD), MOD);
                }
            }
        }
    }

    void bitreversal(std::vector<int> &poly)
    {
        int n = poly.size();
        for (int i = 0, j = 0; i < n; ++i)
        {
            if (i < j)
                std::swap(poly[i], poly[j]);
            int bit = n >> 1;
            while (j & bit)
            {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
        }
    }

    // Inverse NTT
    void INTT(std::vector<int> &poly, const std::vector<int> &roots, int n, int64_t MOD)
    {
        NTT(poly, roots, n, MOD);
        int invN = mod_pow(n, MOD - 2, MOD);
        for (int &x : poly)
            x = mod(1LL * x * invN, MOD);
    }

    // Step 3: Bit field encoding (scaling)
    void bit_field_encode(std::vector<int> &vec, int factor, int64_t MOD)
    {
        int i = 0;
        for (int &val : vec)
        {
            val = mod(val * factor, MOD);
        }
    }

    std::vector<int> encode(const std::vector<int> &input, int n, int64_t MOD, int factor)
    {
        std::vector<int> poly(n, 0);
        for (size_t i = 0; i < input.size(); ++i)
            poly[i] = input[i];
        bit_field_encode(poly, factor, MOD);
        auto roots = precomputeRoots(n, MOD, false);
        NTT(poly, roots, n, MOD);
        return poly; // Return evaluation representation
    }

    int find_primitive_root(int mod)
    {
        int phi = mod - 1; // Euler's totient function value for prime mod is mod-1
        std::vector<int> factors = get_prime_factors_vector(phi);

        for (int g = 2; g < mod; ++g)
        {
            bool is_primitive = true;
            for (int factor : factors)
            {
                // Check if g^((mod-1)/factor) ≡ 1 (mod mod)
                if (mod_pow(g, phi / factor, mod) == 1)
                {
                    is_primitive = false;
                    break;
                }
            }
            if (is_primitive)
                return g; // Return the first primitive root found
        }
        return -1; // Should not reach here for prime mod
    }

    std::pair<int*, int> get_prime_factors(int n)
    {
        std::vector<int> factors;
        for (int i = 2; i * i <= n; ++i)
        {
            while (n % i == 0)
            {
                factors.push_back(i);
                n /= i;
            }
        }
        if (n > 1)
            factors.push_back(n);

        int* arr = (int*)malloc(factors.size() * sizeof(int));
        std::copy(factors.begin(), factors.end(), arr);
        return { arr, factors.size() };
    }

    std::vector<int> get_prime_factors_vector(int n)
    {
        std::vector<int> factors;
        for (int i = 2; i * i <= n; ++i)
        {
            while (n % i == 0)
            {
                factors.push_back(i);
                n /= i;
            }
        }
        if (n > 1)
            factors.push_back(n);
        return factors;
    }
 

    std::vector<int> precomputeRoots(int n, int64_t MOD, bool inverse)
    {
        std::vector<int> roots(n);
        int root = mod_pow(find_primitive_root(MOD), (MOD - 1) / n, MOD);
        if (inverse)
            root = mod_pow(root, MOD - 2, MOD); // Modular inverse of root
        roots[0] = 1;
        for (int i = 1; i < n; ++i)
            roots[i] = mod(roots[i - 1] * root, MOD);

        return roots;
    }

    std::vector<int> decode(const std::vector<int> &poly, int n, int64_t MOD, int factor)
    {
        std::vector<int> coeffs = poly;
        auto roots = precomputeRoots(n, MOD, true);
        INTT(coeffs, roots, n, MOD);
        printf("cpu coeffs: ");
        for (int i = 0; i < n; ++i) printf("%d ", coeffs[i]);
        printf("\n");
        bit_field_encode(coeffs, mod_pow(factor, MOD - 2, MOD), MOD);
        return coeffs; // Return coefficient representation
    }

    void ClearTextEncodingTest()
    {
        int n = 8;                                                                                                                            // Must be a power of 2
        std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7}; // Input vector

        // Encoding
        std::cout << "Input vector: ";
        for (int x : input)
            std::cout << x << " ";
        std::cout << "\n";

        std::vector<int> encoded = encode(input, n, 12289 , 1024);
        std::cout << "Encoded (evaluation form): ";
        for (int x : encoded)
            std::cout << x << " ";
        std::cout << "\n";

        // Decoding
        std::vector<int> decoded = decode(encoded, n, 12289, 1024);
        std::cout << "Decoded (coefficient form): ";
        for (int x : decoded)
            std::cout << x << " ";
        std::cout << "\n";
    }
}