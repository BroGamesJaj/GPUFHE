#include <iostream>
#include <vector>
#include <complex>
#include <cmath>


// Modulus for the finite field
const int MOD = 12289; // Typically a prime modulus used in RLWE schemes
const int N = 8; // Degree of the polynomial, change as needed

// Utility function: Modulo operation
int mod(int x, int mod) {
    return ((x % mod) + mod) % mod;
}

// Utility function: Fast modular exponentiation
int mod_pow(int base, int exp, int modulo) {
    int result = 1;
    base = mod(base, modulo);
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = mod(result * base, modulo);
        }
        base = mod(base * base, modulo);
        exp /= 2;
    }
    return result;
}

// Step 1: Find the primitive root (for simplicity, assuming we know it)
int primitive_root = 3; // This can be any primitive root in the finite field MOD

// Step 2: Perform Number-Theoretic Transform (NTT) for evaluation encoding
void ntt(std::vector<int>& vec, int root, bool inverse = false) {
    int n = vec.size();
    int logn = log2(n);

    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        int rev = 0;
        for (int j = 0; j < logn; j++) {
            if (i & (1 << j)) {
                rev |= (1 << (logn - 1 - j));
            }
        }
        if (rev > i) std::swap(vec[i], vec[rev]);
    }

    // NTT / INTT main loop
    for (int len = 2; len <= n; len <<= 1) {
        int step = mod_pow(root, (MOD - 1) / len, MOD);
        if (inverse) step = mod_pow(step, MOD - 2, MOD); // For INTT, use modular inverse

        for (int i = 0; i < n; i += len) {
            int w = 1;
            for (int j = 0; j < len / 2; j++) {
                int u = vec[i + j];
                int v = (vec[i + j + len / 2] * w) % MOD;
                vec[i + j] = mod(u + v, MOD);
                vec[i + j + len / 2] = mod(u - v, MOD);
                w = (w * step) % MOD;
            }
        }
    }

    if (inverse) {
        // Scaling for INTT
        int inv_n = mod_pow(n, MOD - 2, MOD);
        for (int i = 0; i < n; i++) {
            vec[i] = (vec[i] * inv_n) % MOD;
        }
    }
}

// Step 3: Bit field encoding (scaling)
void bit_field_encode(std::vector<int>& vec, int factor) {
    for (int& val : vec) {
        val = (val * factor) % MOD;
    }
}

// Step 4: Evaluation encoding (coefficient form to evaluation form)
void evaluation_encode(std::vector<int>& coeffs) {
    // Perform NTT to get evaluation points
    ntt(coeffs, primitive_root);
}

// Step 5: Inverse Evaluation encoding (evaluation form to coefficient form)
void inverse_evaluation_encode(std::vector<int>& evals) {
    // Perform INTT to get coefficients
    ntt(evals, primitive_root, true);
}

// Main function to demonstrate the process
int main() {
    // Example coefficients for a polynomial
    std::vector<int> coeffs = {3, 5, 7, 9, 11, 13, 15, 17}; // Polynomial: 3 + 5x + 7x^2 + ... (mod MOD)

    std::cout << "Original Coefficients: ";
    for (int coeff : coeffs) {
        std::cout << coeff << " ";
    }
    std::cout << std::endl;

    // Step 1: Evaluation encoding (coefficient form to evaluation form)
    evaluation_encode(coeffs);

    std::cout << "Evaluation Encoding: ";
    for (int coeff : coeffs) {
        std::cout << coeff << " ";
    }
    std::cout << std::endl;

    // Step 2: Inverse Evaluation encoding (evaluation form to coefficient form)
    inverse_evaluation_encode(coeffs);

    std::cout << "Recovered Coefficients: ";
    for (int coeff : coeffs) {
        std::cout << coeff << " ";
    }
    std::cout << std::endl;

    // Step 3: Apply bit field encoding (scaling the coefficients for noise handling)
    int factor = 128; // Example factor to scale the coefficients
    bit_field_encode(coeffs, factor);

    std::cout << "After Bit Field Encoding: ";
    for (int coeff : coeffs) {
        std::cout << coeff << " ";
    }
    std::cout << std::endl;

    return 0;
}



//primitive root:
std::vector<int> find_divisors(int n) {
    std::vector<int> divisors;
    for (int i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            divisors.push_back(i);
            if (i != n / i) divisors.push_back(n / i);
        }
    }
    return divisors;
}

// Function to check if g is a primitive root modulo p
bool is_primitive_root(int g, int p) {
    int p_minus_1 = p - 1;
    std::vector<int> divisors = find_divisors(p_minus_1);
    
    // Check if g^d % p == 1 for any divisor d of p-1 (except p-1 itself)
    for (int d : divisors) {
        if (d < p_minus_1 && mod_pow(g, d, p) == 1) {
            return false;  // g is not a primitive root
        }
    }
    return true;  // g is a primitive root
}

// Main function to find a primitive root modulo p
int find_primitive_root(int p) {
    for (int g = 2; g < p; g++) {
        if (is_primitive_root(g, p)) {
            return g;  // g is the primitive root
        }
    }
    return -1;  // No primitive root found (should not happen for prime p)
}

int primitiveroot_main() {
    int p = 11;  // Example: p = 11
    
    int g = find_primitive_root(p);
    if (g != -1) {
        std::cout << "Primitive root modulo " << p << " is: " << g << std::endl;
    } else {
        std::cout << "No primitive root found" << std::endl;
    }
    
    return 0;
}
