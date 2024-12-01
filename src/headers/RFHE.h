#pragma once
#include <iostream>
#include "headers/GenArray.h"

class RFHE {
    private:
        int64_t* q;         //maximum number
        int64_t* degree;    // maximum degree

        GenArray* pK;        //private key;
        GenArray* PuK[2];       //public key;

    public:
        RFHE(int64_t q, int64_t degree);
        ~RFHE();

        GenArray genRand(int64_t size);

        void genPK();
        void genPuk();

        GenArray getPK();       //not a good idea
        GenArray** getPuK();

        GenArray addPoly(GenArray array1, GenArray array2);
        GenArray subPoly(GenArray array1, GenArray array2);
        GenArray multPoly(GenArray array1, GenArray array2);
        GenArray redcPoly(GenArray& array);
        GenArray polyMod(GenArray& array,int64_t c = -1);

        void Encrypt(GenArray& message, GenArray** PuK, GenArray*& result);
        GenArray Decrypt(GenArray& c0, GenArray& c1);

        GenArray EncInt(int64_t& message);
        int64_t DecInt(GenArray& encoded);
   
};