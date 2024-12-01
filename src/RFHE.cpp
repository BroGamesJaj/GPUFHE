#include <iostream>
#include <cstdlib>
#include "headers/RFHE.h"
#include <random>
#include <ctime>
#include <chrono>




RFHE::RFHE(int64_t qIn, int64_t degreeIn){
    q = new int64_t;      
    degree = new int64_t;

    pK = new GenArray;
    PuK[0] = new GenArray;
    PuK[1] = new GenArray;

    *q = qIn;
    *degree = degreeIn;
}

RFHE::~RFHE(){
}

GenArray RFHE::genRand(int64_t size) {
    GenArray array(size);

    static std::random_device rd;  
    static std::mt19937_64 rng(rd());
    std::uniform_int_distribution<int64_t> dist(0, *q - 1);

    for (size_t i = 0; i < size; i++) {
        array[i] = dist(rng);
    }

    return array;
}

GenArray RFHE::addPoly(GenArray array1, GenArray array2){

    
    size_t n1 = array1.getSize();
    size_t n2 = array2.getSize();
    size_t n = (n1 > n2) ? n1 : n2;
    GenArray result (n);

    for (size_t i = 0; i < n; i++)
    {
        if(i < n1) result[i] += (result[i]+array1[i]);
        if(i < n2) result[i] += (result[i]+array2[i]);
    }

    return result;
}

GenArray RFHE::subPoly(GenArray array1, GenArray array2){
    GenArray result;
    
    size_t n1 = array1.getSize();
    size_t n2 = array2.getSize();
    size_t n = (n1 > n2) ? n1 : n2;
    result = GenArray(n);

    for (size_t i = 0; i < n; i++)
    {
        if(i < n1) result[i] = (result[i]+array1[i]);
        if(i < n2) result[i] = (result[i]-array2[i]);
    }

    return result;
}


GenArray RFHE::multPoly(GenArray array1, GenArray array2){
    size_t n1 = array1.getSize();
    size_t n2 = array2.getSize();
    size_t n = n1 + n2 - 1;
    GenArray result(n);
    
    for (size_t i = 0; i < n1; i++)
    {
        for (size_t j = 0; j < n2; j++)
        {
            result[i + j] += array1[i] * array2[j];
        }
        
    }

    return result;
}

GenArray RFHE::redcPoly(GenArray& array){
    GenArray result = array;

    while(result.getSize() >= *degree){
        int64_t lastCoeff = result[result.getSize() - 1];

        for (int i = 0; i < *degree; ++i) {
            result[result.getSize() - 1 - i] = (result[result.getSize() - 1 - i] - lastCoeff) % *q;
        }

        while (result.getSize() > 0 && result.back() == 0)
        {
            result.pop_back();
        }  
    }

    return result;
}

void RFHE::genPK(){
    *pK = genRand((*degree)-1);
}

void RFHE::genPuk(){
    GenArray a = genRand(*degree); //a as alap
    GenArray e = genRand(*degree); //e as everage -> smol

    GenArray b = multPoly(a, *pK);
    b = addPoly(b, e);
    b = redcPoly(b);

    *PuK[0] = a;
    *PuK[1] = b;
}

GenArray RFHE::getPK(){
    return *pK;
}

GenArray** RFHE::getPuK(){
    return PuK;
}

void RFHE::Encrypt(GenArray& message, GenArray** PuK, GenArray*& result) {
    GenArray a = genRand(*degree);
    GenArray e1 = genRand(*degree);
    GenArray e2 = genRand(*degree);

    GenArray c0 = addPoly(multPoly(*PuK[1], a), e1);
    GenArray c1 = addPoly(addPoly(multPoly(*PuK[0], a), message), e2);

    c0 = polyMod(c0);
    c1 = polyMod(c1);

    result[0] = c0;
    result[1] = c1;
}

GenArray RFHE::Decrypt(GenArray& c0, GenArray& c1) {
    GenArray vmi = multPoly(c1, *pK);

    GenArray decrypted = subPoly(c0, vmi);
    decrypted = redcPoly(decrypted);

    for (size_t i = 0; i < decrypted.getSize(); ++i) {
        decrypted[i] = (decrypted[i] + *q) % *q;
    }
    return decrypted;
}

GenArray RFHE::EncInt(int64_t& message) {
    GenArray encoded;

    while (message > 0) {
        encoded.PutLast(message % *q);
        message /= *q;
    }

    return encoded;
}

int64_t RFHE::DecInt(GenArray& encoded) {
    int64_t message = 0;
    int64_t multiplier = 1;

    for (size_t i = 0; i < encoded.getSize(); ++i) {
        message += encoded[i] * multiplier;
        multiplier *= *q;
    }

    return message;
}

GenArray RFHE::polyMod(GenArray& array, int64_t c){
    int64_t size = array.getSize();
    GenArray result(size);

    if(c == -1) {
        for (size_t i = 0; i < size; i++)
        {
            result[i] = (array[i] % *q + *q) % *q;
        }
    } else{
        for (size_t i = 0; i < size; i++)
        {
            result[i] = (array[i] % c + c) % c;
        }
    }

    return result;
}
