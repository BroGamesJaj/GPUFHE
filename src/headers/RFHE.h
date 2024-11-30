#pragma once
#include <iostream>
#include "headers/GenArray.h"

class RFHE {
    private:
        int64_t* q;         //maximum number
        int64_t* degree;    // maximum degree

        GenArray* pK;        //private key;
        GenArray* PuK;       //public key;

    public:
        RFHE(int64_t& q, int64_t& degree);
        ~RFHE();
        
        /*
        void setQ();
        void setDegree();
        */

        GenArray genRand(size_t& size);

        void genPK();
        void genPuk();

        void getPK();       //not a good idea
        void getPuK();

        GenArray addPoly(GenArray& array1, GenArray& array2);
        GenArray* multPoly();
};