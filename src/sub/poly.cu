#include "poly.h"

namespace poly{

    GeneralArray<int64_t> initPolyModulus(int poly_mod){
        GeneralArray<int64_t> poly_modulus(poly_mod+1);
        poly_modulus[0] = 1;
        poly_modulus[poly_mod] = 1;

        return poly_modulus;
    }

    GeneralArray<int64_t> initPolyModulus(GeneralArray<int64_t> poly_mod){
        GeneralArray<int64_t> polyModulus(poly_mod.getSize());
        for (size_t i = 0; i < poly_mod.getSize(); ++i) {
            polyModulus[i] = poly_mod[i];
        }
        return polyModulus;
    }

    GeneralArray<int64_t> PolyMod(GeneralArray<int64_t> poly_array, int64_t c){
        for (size_t i = 0; i < poly_array.getSize(); i++) {
            poly_array.getArray()[i] %= c;
        }
        return poly_array;
    }

    GeneralArray<int64_t> modCenter(GeneralArray<int64_t>& poly, int64_t coeff_modulus, bool left_closed) {
        if (coeff_modulus <= 0) {
            throw std::invalid_argument("Coefficient modulus must be a positive integer.");
        }
        for (size_t i = 0; i < poly.getSize(); ++i) {
            if (left_closed) {
                poly[i] = ((poly[i] + coeff_modulus / 2) % coeff_modulus + coeff_modulus) % coeff_modulus - coeff_modulus / 2;

            } else {
                poly[i] = ((poly[i] + coeff_modulus / 2 - 1) % coeff_modulus + coeff_modulus) % coeff_modulus - coeff_modulus / 2 + 1;
            }
        }
        return poly;
    }

    std::pair<GeneralArray<int64_t>, GeneralArray<int64_t>> PolyDiv_cpu_ga(const GeneralArray<int64_t>& dividend, GeneralArray<int64_t> divisor) {
        //std::cout << "PolyDiv ga started " << std::endl;
        // Remove trailing zeros from the divisor
        while (divisor[divisor.getSize() - 1] == 0) {
            divisor.pop_back();
            divisor.print();
        }
        if (divisor.getSize() < 2) {
            throw std::runtime_error("Divisor must be at least degree 1");
        }
        size_t dividendSize = dividend.getSize();
        size_t divisorSize = divisor.getSize();
        // Initialize quotient and remainder arrays
        GeneralArray<int64_t> quotient(dividendSize - divisorSize + 1);
        GeneralArray<int64_t> remainder(dividendSize);
        remainder = dividend;  // Start with the dividend as the remainder
        for (int i = dividendSize - 1; i >= divisorSize - 1; --i) {
            int64_t coeff_div = remainder[i] / divisor[divisorSize - 1];
            quotient[i - divisorSize + 1] = coeff_div;
            for (int j = 0; j < divisorSize; ++j) {
                remainder[i - j] -= coeff_div * divisor[divisorSize - 1 - j];
            }
        }
        // Return the quotient and remainder as a pair
        return {quotient, remainder};
    }

    Polinomial randomTernaryPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus) {
        Polinomial result(poly_modulus.getSize(), coeff_modulus, poly_modulus);
        
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 0; i < result.getSize(); ++i) {
            double randVal = dist(rng);
            
            if (randVal < 0.25) {
                result[i] = -1;
            } else if (randVal < 0.75) {
                result[i] = 0;
            } else {
                result[i] = 1;
            }
            
        }
        return result;
    }

    Polinomial randomTernaryPoly(int64_t coeff_modulus, const int64_t poly_modulus) {
        Polinomial result(poly_modulus, coeff_modulus, poly_modulus);
        
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 0; i < result.getSize(); ++i) {
            double randVal = dist(rng);
            
            if (randVal < 0.25) {
                result[i] = -1;
            } else if (randVal < 0.75) {
                result[i] = 0;
            } else {
                result[i] = 1;
            }

        }
        return result;
    }

    Polinomial randomBinaryPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus){
        Polinomial result(poly_modulus.getSize(), coeff_modulus, poly_modulus);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> dist(0, 1);

        for (size_t i = 0; i < poly_modulus.getSize(); ++i) {
            result[i] = dist(rng);
        }

        return result;
    }
    
    Polinomial randomBinaryPoly(int64_t coeff_modulus, const int64_t poly_modulus){
        Polinomial result(poly_modulus, coeff_modulus, poly_modulus);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> dist(0, 1);

        for (size_t i = 0; i < result.getSize(); ++i) {
            result[i] = dist(rng);
        }

        return result;
    }

    Polinomial randomUniformPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus, int64_t high){
        if (high == -1) {
            high = coeff_modulus/4 - 1;
        }
        int64_t offset = -high/2;
        Polinomial result(poly_modulus.getSize(), coeff_modulus, poly_modulus);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int64_t> dist(0 + offset, high-1 + offset);
        for (size_t i = 0; i < result.getSize(); ++i) {
            result[i] = dist(rng);
        }
        result.modCenter();
        return result;
    }

    Polinomial randomUniformPoly(int64_t coeff_modulus, const int64_t poly_modulus, int64_t high){
        if (high == -1) {
            high = coeff_modulus/2 - 1;
        }
        int64_t offset = -high/2;
        Polinomial result(poly_modulus, coeff_modulus, poly_modulus);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int64_t> dist(0 + offset, high-1 + offset);
        for (size_t i = 0; i < result.getSize(); ++i) {
            result[i] = dist(rng);
        }
        result.modCenter();
        return result;
    }

    Polinomial randomUniformPolyMSG(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus, int64_t high, int64_t max_degree){
        if (high == -1) {
            high = coeff_modulus - 1;
        }

        int64_t offset = -high/2;
        Polinomial result(poly_modulus.getSize(), coeff_modulus, poly_modulus);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int64_t> dist(0 + offset, high-1 + offset);
        for (size_t i = 0; i < result.getSize(); ++i) {
            result[i] = dist(rng);
        }
        result.modCenter(high);
        return result;
    }

    Polinomial randomNormalPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus, double mean, double std){

        Polinomial result(poly_modulus.getSize(), coeff_modulus, poly_modulus);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(mean, std);

        for (size_t i = 0; i < result.getSize(); ++i) {
            double rand_val = dist(gen);

            // Round and clamp to a reasonable range, e.g., [-kσ, kσ] for some small k (e.g., k = 6)
            double clamped_val = std::max(std::min(rand_val, mean + 6 * std), mean - 6 * std);

            result[i] = static_cast<int64_t>(std::round(clamped_val));
        }
        result.modCenter();
        return result;
    }

    Polinomial randomNormalPoly(int64_t coeff_modulus, const int64_t poly_modulus, double mean, double std){

        Polinomial result(poly_modulus, coeff_modulus, poly_modulus);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(mean, std);

        for (size_t i = 0; i < result.getSize(); ++i) {
            double rand_val = dist(gen);
            result[i] = std::round(rand_val);
        }
        result.modCenter();

        return result;
    }

    Polinomial discreteGaussianSampler(int64_t coeff_modulus, const GeneralArray<int64_t>& poly_modulus, double sigma) {
        // Initialize the resulting polynomial
        Polinomial result(poly_modulus.getSize() - 1, coeff_modulus, poly_modulus);

        // Random number generators
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        std::normal_distribution<double> gaussian_dist(0.0, sigma);

        for (size_t i = 0; i < result.getSize(); ++i) {
            bool accepted = false;
            while (!accepted) {
                // Sample a candidate from a normal distribution
                double candidate = gaussian_dist(generator);

                // Round to the nearest integer
                int64_t rounded_candidate = static_cast<int64_t>(std::round(candidate));

                // Compute rejection probability
                double acceptance_prob = std::exp(-std::pow(candidate - rounded_candidate, 2) / (2.0 * sigma * sigma));

                // Accept or reject the candidate
                if (uniform_dist(generator) < acceptance_prob) {
                    // Ensure the value is in the range [0, coeff_modulus)
                    int64_t modded_candidate = (rounded_candidate % coeff_modulus + coeff_modulus) % coeff_modulus;
                    result[i] = modded_candidate;
                    accepted = true;
                }
            }
        }

        return result;
    }

}