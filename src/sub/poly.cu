#include "poly.h"

namespace poly{

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
            high = coeff_modulus - 1;
        }

        Polinomial result(poly_modulus.getSize(), coeff_modulus, poly_modulus);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int64_t> dist(0, high);
        for (size_t i = 0; i < result.getSize(); ++i) {
            result[i] = dist(rng);
        }

        return result;
    }

    Polinomial randomUniformPoly(int64_t coeff_modulus, const int64_t poly_modulus, int64_t high){
        if (high == -1) {
            high = coeff_modulus - 1;
        }

        Polinomial result(poly_modulus, coeff_modulus, poly_modulus);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int64_t> dist(0, high);
        for (size_t i = 0; i < result.getSize(); ++i) {
            result[i] = dist(rng);
        }

        return result;
    }

    Polinomial randomNormalPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus, double mean, double std){

        Polinomial result(poly_modulus.getSize(), coeff_modulus, poly_modulus);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(mean, std);

        for (size_t i = 0; i < result.getSize(); ++i) {
            double rand_val = dist(gen);
            result[i] = std::round(rand_val);

            result[i] = result[i] % coeff_modulus;
            if (result[i] < 0) {
                result[i] += coeff_modulus;
            }
        }
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

            result[i] = result[i] % coeff_modulus;
            if (result[i] < 0) {
                result[i] += coeff_modulus;
            }
        }
        return result;
    }
}