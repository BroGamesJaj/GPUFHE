#include "start.cuh"

int main(){

    //cleartext_encoding::ClearTextEncodingTest();
    printf("started\n");
    int64_t n = 2048; // degree of the polynomials
    int64_t coef_modulus = pow(2,44); // can the second value if you want to change the size of q(coefficient_modulus)
    int64_t plaintext_modulus = pow(2,32); // max size of stored values (max 32 if no operations on poly)
    int64_t max_degree = 16; // amount of numbers stored
    int base = 12;
    GeneralArray<int64_t> poly_modulus = poly::initPolyModulus(n); 
    
    Polinomial sk = bgvfhe_gpu::GeneratePrivateKey(coef_modulus, poly_modulus);
    auto pk = bgvfhe_gpu::GeneratePublicKey(sk, coef_modulus, poly_modulus, plaintext_modulus);

    tests::cAddTest(n,coef_modulus,plaintext_modulus,poly_modulus,sk,pk,10);
    tests::cSubTest(n,coef_modulus,plaintext_modulus,poly_modulus,sk,pk,10);

    tests::cMultTest(n,coef_modulus,plaintext_modulus,poly_modulus,sk,pk,2,10);
    return 0;
}