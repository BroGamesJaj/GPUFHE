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

            size_t getSize() const { return coeff.get_size(); }

            GeneralArray<uint64_t>& getCoeff() { return coeff; }

            uint64_t& operator[](size_t index) { return coeff[index]; }
    };
 
}
    
using poly::Polinomial;