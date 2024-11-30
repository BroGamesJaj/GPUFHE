#include <iostream>
#include <cstdlib>
#include "headers/RFHE.h"
#include <random>
#include <ctime>




RFHE::RFHE(int64_t& q, int64_t& degree){
    q = q;
    degree = degree;
}

RFHE::~RFHE(){
}

GenArray RFHE::genRand(size_t& size){
    GenArray array(size);

    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<int64_t> dist(0,*q-1);

    for (size_t i = 0; i < size; i++)
    {
        array[i] = dist(rng);
    }

    return array;
}

GenArray RFHE::addPoly(GenArray& array1, GenArray& array2){
    GenArray result;
    
    size_t n1 = array1.getSize();
    size_t n2 = array2.getSize();
    size_t n = (n1 > n2) ? n1 : n2;
    result = GenArray(n);

    for (size_t i = 0; i < n; i++)
    {
        if(i < n1) result[i] = (result[i]+array1[i]) % *q;
        if(i < n2) result[i] = (result[i]+array2[i]) % *q;
    }

    return result;
}

