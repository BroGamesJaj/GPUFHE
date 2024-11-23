#pragma once
#include "general_array.h"
#include <cstdint>

namespace poly {
    class Polinomial{
        private:
            size_t size;
            general_array::GeneralArray<uint64_t> coeff;
    };


}
    
