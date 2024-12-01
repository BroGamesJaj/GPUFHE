#pragma once

#include <cstdint>
#include <random>
#include "general_array.h"

namespace poly {

    GeneralArray<int64_t> initPolyModulus(int poly_mod);
    GeneralArray<int64_t> initPolyModulus(GeneralArray<int64_t> poly_mod);
    GeneralArray<int64_t> PolyMod(GeneralArray<int64_t> poly_array, int64_t c);

    GeneralArray<int64_t> modCenter(GeneralArray<int64_t>& poly, int64_t coeff_modulus, bool left_closed = true);

    std::pair<GeneralArray<int64_t>, GeneralArray<int64_t>> PolyDiv_cpu_ga(const GeneralArray<int64_t>& dividend, GeneralArray<int64_t> divisor);

    class Polinomial{
        private:
            GeneralArray<int64_t> coeff;
            int64_t coeff_modulus;
            GeneralArray<int64_t> poly_modulus;
        public:

            Polinomial(size_t initialSize) : coeff(initialSize), coeff_modulus(0), poly_modulus(0){}
            
            Polinomial(size_t initialSize, int64_t modulus) : coeff(initialSize), coeff_modulus(modulus), poly_modulus(0){}

            Polinomial(size_t initialSize, int64_t* array) : coeff(initialSize, array), coeff_modulus(0), poly_modulus(0){}

            Polinomial(size_t initialSize, int64_t modulus, int64_t polyMod) : coeff(polyMod-1), coeff_modulus(modulus), poly_modulus(initPolyModulus(polyMod)){
                reducePolynomial();
            }

            Polinomial(size_t initialSize, int64_t modulus, GeneralArray<int64_t> polyMod) : coeff(polyMod.getSize()-1), coeff_modulus(modulus), poly_modulus(initPolyModulus(polyMod)){
                reducePolynomial();
            }

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

            Polinomial operator-() const {
                Polinomial negatedPoly(coeff.getSize(), coeff_modulus, poly_modulus);
                for (size_t i = 0; i < coeff.getSize(); i++) {
                    negatedPoly[i] = -coeff[i];
                }
                return negatedPoly;
            }

            bool operator==(const Polinomial& other) const {
                if (coeff.getSize() != other.getSize()) {
                    return false;
                }
                for (size_t i = 0; i < coeff.getSize(); ++i) {
                    if (coeff[i] != other[i]) {
                        return false;
                    }
                }
                return true;
            }

            void untrim(){
                if(getSize() < getPolyModSize()){
                    getCoeff().resize(getPolyModSize()-1);
                }
            }

            void polyMod(int64_t c = -1){
                if(c == -1) {
                    for (int64_t& coef : getCoeff()) {
                        coef = (coef % getCoeffModulus() + getCoeffModulus()) % getCoeffModulus();
                    }
                } else{
                    for (int64_t& coef : getCoeff()) {
                        coef = (coef % c + c) % c;
                    }
                }
                
            }
            void reducePolynomial(int64_t modulo = -1) {
                if(modulo == -1){
                    polyMod();
                    while(getSize() >= getPolyModSize() && getSize() > 0){

                        int64_t lastCoeff = getCoeff()[getSize() - 1];
                        for (size_t i = 0; i < getSize()-1; i++) {
                            getCoeff()[i] -= lastCoeff;
                        }
                        getCoeff().pop_back();
                    }
                    polyMod();
                } else{
                    polyMod(modulo);
                    while(getSize() >= getPolyModSize() && getSize() > 0){

                        int64_t lastCoeff = getCoeff()[getSize() - 1];
                        for (size_t i = 0; i < getSize()-1; i++) {
                            getCoeff()[i] -= lastCoeff;
                        }
                        getCoeff().pop_back();
                    }
                    polyMod(modulo);
                }
                
                untrim();
            }

            
            

            void trim() {
                size_t degree = coeff.getSize() - 1;
                while (degree > 0 && coeff[degree] == 0) {
                    --degree;
                }
                coeff.resize(degree + 1);
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
                        } else if (i != 0) {
                            std::cout << " - ";
                        }
                    } else if( i == 0) {
                        if(coeff[i] >= 0){
                            std::cout << "+";
                        }
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
                std::cout << std::endl;
            }
    };

    Polinomial randomTernaryPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus);
    Polinomial randomTernaryPoly(int64_t coeff_modulus, const int64_t poly_modulus);

    Polinomial randomBinaryPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus);
    Polinomial randomBinaryPoly(int64_t coeff_modulus, const int64_t poly_modulus);

    Polinomial randomUniformPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus, int64_t high=-1);
    Polinomial randomUniformPoly(int64_t coeff_modulus, const int64_t poly_modulus, int64_t high=-1);

    Polinomial randomNormalPoly(int64_t coeff_modulus, const GeneralArray<int64_t> poly_modulus, double mean = 0, double std = 3.8);
    Polinomial randomNormalPoly(int64_t coeff_modulus, const int64_t poly_modulus, double mean = 0, double std = 3.8);

    Polinomial discreteGaussianSampler(int64_t coeff_modulus, const GeneralArray<int64_t>& poly_modulus, double sigma = 3.2);
}
    
using poly::Polinomial;