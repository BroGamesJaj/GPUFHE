#pragma once
#include "general_array.h"
#include <cstdint>
#include <random>

namespace poly {

    GeneralArray<int64_t> initPolyModulus(int poly_mod);
    GeneralArray<int64_t> initPolyModulus(const GeneralArray<int64_t> &poly_mod);

    class Polinomial{
        private:
            GeneralArray<int64_t> coeff;
            int64_t coeff_modulus;
            GeneralArray<int64_t> poly_modulus;
        public:

            Polinomial(size_t initialSize) : coeff(initialSize), coeff_modulus(0), poly_modulus(0){}
            
            Polinomial(size_t initialSize, int64_t modulus) : coeff(initialSize), coeff_modulus(modulus), poly_modulus(0){}

            Polinomial(size_t initialSize, int64_t modulus, int64_t polyMod) : coeff(initialSize), coeff_modulus(modulus), poly_modulus(initPolyModulus(polyMod)){}

            Polinomial(size_t initialSize, int64_t modulus, GeneralArray<int64_t> polyMod) : coeff(initialSize), coeff_modulus(modulus), poly_modulus(initPolyModulus(polyMod)){}

            GeneralArray<int64_t> getCoeff() const { return coeff; }

            size_t getSize() const { return coeff.getSize(); }

            GeneralArray<int64_t>& getCoeff() { return coeff; }

            int64_t* getCoeffPointer() const { return coeff.getArray(); }

            int64_t getCoeffModulus() const { return coeff_modulus; }

            GeneralArray<int64_t> getPolyModulus() const { return poly_modulus; }

            size_t getPolyModSize() const { return poly_modulus.getSize(); }

            void setCoeffModulus(int64_t modulus) { coeff_modulus = modulus; }

            void setPolyModulus(const GeneralArray<int64_t>& polyMod) { poly_modulus = polyMod; }

            int64_t& operator[](size_t index) { return coeff[index]; }
            const int64_t& operator[](size_t index) const { return coeff[index]; }

            void reduceCoefficients() {
                if (coeff_modulus == 0) {
                    throw std::runtime_error("Coefficient modulus is not set.");
                }
                for (size_t i = 0; i < coeff.getSize(); ++i) {
                    coeff[i] %= coeff_modulus;
                }
            }
            
            void reducePolynomial() {
                if (poly_modulus.getSize() == 0) {
                    throw std::runtime_error("Polynomial modulus is not set.");
                }

                size_t polyDegree = poly_modulus.getSize() - 1;
                size_t degree = coeff.getSize() - 1;

                while (degree >= polyDegree && coeff[degree] != 0) {
                    uint64_t factor = coeff[degree];
                    for (size_t i = 0; i <= polyDegree; ++i) {
                        if (degree - polyDegree + i < coeff.getSize()) {
                            coeff[degree - polyDegree + i] = 
                                (coeff[degree - polyDegree + i] + coeff_modulus - (factor * poly_modulus[i]) % coeff_modulus) % coeff_modulus;
                        }
                    }
                    --degree;
                }
            }

            void checkQRing(const Polinomial& other) const {
                if (this->poly_modulus.getSize() != other.poly_modulus.getSize()) {
                    throw std::invalid_argument("Polynomials are not in the same Quotient Ring");
                }
                
                for (size_t i = 0; i < this->poly_modulus.getSize(); ++i) {
                    if (this->poly_modulus[i] != other.poly_modulus[i]) {
                        throw std::invalid_argument("Polynomials are not in the same Quotient Ring");
                    }
                }

                if (this->coeff_modulus != other.coeff_modulus) {
                    throw std::invalid_argument("Polynomials are not in the same Quotient Ring");
                }
            }

            void print() const {
                for (size_t i = 0; i < coeff.getSize(); ++i) {
                    if (i > 0) {
                        if (coeff[i] >= 0) {
                            std::cout << " + ";
                        } else {
                            std::cout << " - ";
                        }
                    } else if (coeff[i] < 0) {
                        std::cout << "-";
                    }
                    if (i > 0) {
                        std::cout << std::abs(coeff[i]);
                    } else {
                        std::cout << coeff[i];
                    }
                    if (i > 0) {
                        std::cout << "x^" << i;
                    }
                }
                std::cout << std::endl; // End the line after printing all terms
            }
    };

    Polinomial randomTernaryPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus);
    Polinomial randomTernaryPoly(int64_t coeff_modulus, const int64_t poly_modulus);

    


}
    
using poly::Polinomial;