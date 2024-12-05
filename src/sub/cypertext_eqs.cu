#include "cypertext_eqs.h"

namespace cypertext_eqs{
    std::pair<Polinomial, Polinomial> cAdd_cpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2){
        Polinomial temp1 = poly_eqs::PolyAdd_cpu(e_msg1.first,e_msg2.first);
        temp1.modCenter();
        Polinomial temp2 = poly_eqs::PolyAdd_cpu(e_msg1.second,e_msg2.second);
        temp1.modCenter();
        return std::make_pair(temp1,temp2);
    }

    std::pair<Polinomial, Polinomial> cSub_cpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2){
        Polinomial temp1 = poly_eqs::PolyAdd_cpu(e_msg1.first,-e_msg2.first);
        temp1.modCenter();
        Polinomial temp2 = poly_eqs::PolyAdd_cpu(e_msg1.second,-e_msg2.second);
        temp1.modCenter();
        return std::make_pair(temp1,temp2);
    }

    struct result cMult_cpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2){
        Polinomial temp1 = poly_eqs::PolyMult_cpu(e_msg1.first,e_msg2.first);
        Polinomial temp2 = poly_eqs::PolyAdd_cpu(poly_eqs::PolyMult_cpu(e_msg1.first,e_msg2.second),poly_eqs::PolyMult_cpu(e_msg1.second,e_msg2.first));
        temp2.modCenter();
        Polinomial temp3 = poly_eqs::PolyMult_cpu(e_msg1.second,e_msg2.second);
        return {temp1,temp2,temp3};
    }

}