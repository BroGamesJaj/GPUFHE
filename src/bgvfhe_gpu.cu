#include "bgvfhe_gpu.cuh"
#define N 10



double get_time() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

bool computeNoiseNorm(const Polinomial& poly) {

    int size = poly.getSize();  // Get the size of the polynomial

    int max_noise = 0;
    for (int i = 0; i < size; ++i) {
        if(poly[i] > max_noise) max_noise = poly[i];
    }
    if(max_noise < poly.getCoeffModulus() / 2){
        printf("noise okay\n");
    }else{
        printf("noise bad\n");
    }
    return max_noise < poly.getCoeffModulus() / 2;  // L2 norm (Euclidean norm)
}


Polinomial GeneratePrivateKey(int64_t coeff_modulus, GeneralArray<int64_t> poly_modulus){
    if(coeff_modulus != 0 && poly_modulus.getSize() != 0){
        Polinomial randomPoly = poly::randomTernaryPoly(coeff_modulus, poly_modulus);
        return randomPoly;
    }else{
        throw std::runtime_error("coefficient or poly_modulus is not set");
    }
}

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

auto cMult_cpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2){
    Polinomial temp1 = poly_eqs::PolyMult_cpu(e_msg1.first,e_msg2.first);
    Polinomial temp2 = poly_eqs::PolyAdd_cpu(poly_eqs::PolyMult_cpu(e_msg1.first,e_msg2.second),poly_eqs::PolyMult_cpu(e_msg1.first,e_msg2.second));
    temp2.modCenter();
    Polinomial temp3 = poly_eqs::PolyMult_cpu(e_msg1.second,e_msg2.second);
    struct result {Polinomial c0; Polinomial c1; Polinomial c2;};
    return result {temp1,temp2,temp3};
}

std::pair<Polinomial,Polinomial> GeneratePublicKey(Polinomial& sk, int64_t coeff_modulus, GeneralArray<int64_t>& poly_modulus, int64_t plaintext_modulus){
    Polinomial e = poly::randomNormalPoly(coeff_modulus,poly_modulus);
    //printf("e noise: ");
    //computeNoiseNorm(e);
    Polinomial a = poly::randomTernaryPoly(coeff_modulus,poly_modulus);

    Polinomial temp1 = poly_eqs::PolyMult_cpu(a,sk);
    //printf("temp1 noise: ");
    //computeNoiseNorm(temp1);

    Polinomial temp2 = poly_eqs::PolyMult_cpu(e,plaintext_modulus);
    //printf("temp2 noise: ");
    //computeNoiseNorm(temp2);

    Polinomial b = poly_eqs::PolyAdd_cpu(temp1,temp2);
    computeNoiseNorm(b);
    b.modCenter();

    //PublicKeyTest(b,a,sk,a,e,plaintext_modulus);
    return std::make_pair(b,a);
}



bool isSmallNorm(const Polinomial& poly, int64_t bound) {
    for (int64_t coef : poly.getCoeff()) { // Iterate over coefficients of the polynomial
        if (std::abs(coef) > bound) {
            return false; // Coefficient exceeds the allowed bound
        }
    }
    return true; // All coefficients are within the bound
}

std::pair<Polinomial, Polinomial> asymetricEncryption(Polinomial pk0, Polinomial pk1, Polinomial msg, int64_t plaintext_modulus, int64_t coef_modulus, GeneralArray<int64_t> poly_modulus, int64_t degree){
    Polinomial u = poly::randomTernaryPoly(coef_modulus,poly_modulus);
    Polinomial e0 = poly::randomNormalPoly(coef_modulus,poly_modulus,coef_modulus/static_cast<int>(pow(2, degree)));
    Polinomial e1 = poly::randomNormalPoly(coef_modulus,poly_modulus);
    Polinomial c0_temp1 = poly_eqs::PolyMult_cpu(pk0,u);

    Polinomial c0_temp2 = poly_eqs::PolyMult_cpu(e0,plaintext_modulus);

    Polinomial c0 = poly_eqs::PolyAdd_cpu(poly_eqs::PolyAdd_cpu(c0_temp1,c0_temp2),msg);
    //printf("c0 noise: ");
    //computeNoiseNorm(c0);

    Polinomial c1_temp1 = poly_eqs::PolyMult_cpu(pk1,u);

    Polinomial c1_temp2 = poly_eqs::PolyMult_cpu(e1,plaintext_modulus);

    Polinomial c1 = poly_eqs::PolyAdd_cpu(c1_temp1,c1_temp2);
    //printf("c1 noise: ");
    //computeNoiseNorm(c1);

    //printf("c0\n");
    //c0.print();
    //c1.print();
    return std::make_pair(c0,c1);
}

Polinomial decrypt(Polinomial c0, Polinomial c1, Polinomial sk, int64_t plaintext_modulus){
    Polinomial sk_c1 = poly_eqs::PolyMult_cpu(c1,sk);
    //printf("sk_c1 noise: ");
    //computeNoiseNorm(sk_c1);
    Polinomial msg = poly_eqs::PolySub_cpu(c0,sk_c1);
    //computeNoiseNorm(msg);

    msg.modCenter(plaintext_modulus);
    
    return msg;
}



bool isNoiseSmallEnough(const Polinomial& noise, double threshold) {
    double norm = computeNoiseNorm(noise);
    std::cout << norm << std::endl;
    return norm < threshold;  // Check if the noise norm is below the threshold
}


int main(){

    double start_time = get_time();
    //cleartext_encoding::ClearTextEncodingTest();
    printf("started\n");
    int64_t n = 2048; // degree of the polynomials
    int64_t coef_modulus = pow(2,40); // can the second value if you want to change the size of q(coefficient_modulus)
    int64_t plaintext_modulus = pow(2,30); // max size of stored values (max 32 if no operations on poly)
    int64_t max_degree = 500; // amount of numbers stored
    for (size_t i = 0; i < 5; i++) {}
    GeneralArray<int64_t> poly_modulus = poly::initPolyModulus(n); 

    
    Polinomial sk = GeneratePrivateKey(coef_modulus, poly_modulus);
    auto pk = GeneratePublicKey(sk, coef_modulus, poly_modulus, plaintext_modulus);
    printf("PK generator ended\n");

    Polinomial msg = poly::randomUniformPolyMSG(coef_modulus, poly_modulus, plaintext_modulus/8,max_degree > n ? n : max_degree);
    Polinomial msg2 = poly::randomUniformPolyMSG(coef_modulus, poly_modulus, plaintext_modulus/8,max_degree > n ? n : max_degree);
    printf("MSG:\n");
    msg.print();
    printf("MSG2:\n");
    msg2.print();
    auto e_msg = asymetricEncryption(pk.first,pk.second,msg,plaintext_modulus,coef_modulus,poly_modulus,n);
    auto e_msg2 = asymetricEncryption(pk.first,pk.second,msg2,plaintext_modulus,coef_modulus,poly_modulus,n);

    Polinomial decrypted_msg = decrypt(e_msg.first, e_msg.second,sk,plaintext_modulus);
    printf("decrypted MSG1:\n");
    decrypted_msg.print();
    Polinomial decrypted_msg2 = decrypt(e_msg2.first, e_msg2.second,sk,plaintext_modulus);
    printf("decrypted MSG2:\n");
    decrypted_msg2.print();
    if (decrypted_msg == msg) {
        printf("Decryption successful\n");
    } else {
        printf("Decryption failed\n");
    }
    std::cout << (coef_modulus >> 1) << std::endl;
    if(isNoiseSmallEnough(e_msg.first,coef_modulus >> 1)){
        printf("Good noise\n");
    } else {
        printf("Bad noise\n");
    }
    double end_time = get_time();
    printf("CPU run time: %f milliseconds\n", (end_time - start_time)*1000);

}