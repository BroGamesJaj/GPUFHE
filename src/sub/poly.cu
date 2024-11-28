#include "poly.h"

namespace poly{

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
            
            result[i] %= coeff_modulus;
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
            
            result[i] %= coeff_modulus;
        }
        return result;
    }

    GeneralArray<int64_t> initPolyModulus(int poly_mod){
        GeneralArray<int64_t> poly_modulus(poly_mod);
        poly_modulus[0] = 1;
        poly_modulus[poly_mod - 1] = 1;

        return poly_modulus;
    }

    GeneralArray<int64_t> initPolyModulus(const GeneralArray<int64_t> &poly_mod){
        GeneralArray<int64_t> polyModulus(poly_mod.getSize());
        for (size_t i = 0; i < poly_mod.getSize(); ++i) {
            polyModulus[i] = poly_mod[i];
        }
        return polyModulus;
    }
}