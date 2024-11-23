#include "poly_eqs.h"

namespace poly_eqs{
    Polinomial PolyMult(Polinomial p1, Polinomial p2){
        Polinomial prod(p1.getSize()+p2.getSize());

        for (int i=0; i<p1.getSize(); i++) { 
            for (int j=0; j<p2.getSize(); j++){
                prod[i+j] += p1[i]*p2[j]; 
            }
        } 

        return prod;
    }

}