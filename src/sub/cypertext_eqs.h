#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "poly.h"
#include "poly_eqs.h"
#include <cuda_runtime.h>

namespace cypertext_eqs{
    struct result {Polinomial c0; Polinomial c1; Polinomial c2;};
    std::pair<Polinomial, Polinomial> cAdd_cpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2);

    std::pair<Polinomial, Polinomial> cSub_cpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2);

    struct result cMult_cpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2);
}