#pragma once
#include "general_array.h"
#include <cstdint>

namespace poly {
    class Polinomial{
        private:
            GeneralArray<uint64_t> coeff;

        public:

            Polinomial(size_t initialSize) : coeff(initialSize){
            }

            GeneralArray<uint64_t> getCoeff() const { return coeff; }

            size_t getSize() const { return coeff.getSize(); }

            GeneralArray<uint64_t>& getCoeff() { return coeff; }

            uint64_t* getCoeffPointer() const { return coeff.getArray(); }

            uint64_t& operator[](size_t index) { return coeff[index]; }

            uint64_t& operator[](size_t index) const { return coeff.getArray()[index]; }

            uint64_t back() { coeff.back(); }

            void pop_back() { coeff.pop_back(); }
        
            void print() const {
                for (size_t i = 0; i < coeff.getSize(); ++i) {
                    if (i != 0 && coeff[i] > 0) std::cout << "+";
                    std::cout << coeff[i];
                    if (i < coeff.getSize() - 1) std::cout << "x^" << (coeff.getSize() - i - 1) << " ";
                }
                std::cout << std::endl;
            }
    };
 
}
    
using poly::Polinomial;