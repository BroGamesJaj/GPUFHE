#include "bgvfhe_gpu.cuh"
#define N 10

namespace bgvfhe_gpu{
        
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

    std::pair<Polinomial,Polinomial> GeneratePublicKey(Polinomial& sk, int64_t coeff_modulus, GeneralArray<int64_t>& poly_modulus, int64_t plaintext_modulus){
        Polinomial e = poly::randomNormalPoly(coeff_modulus,poly_modulus);
        Polinomial a = poly::randomTernaryPoly(coeff_modulus,poly_modulus);
        Polinomial temp1 = poly_eqs::PolyMult_cpu(a,sk);
        Polinomial temp2 = poly_eqs::PolyMult_cpu(e,plaintext_modulus);

        Polinomial b = poly_eqs::PolyAdd_cpu(temp1,temp2);

        b.modCenter();

        return std::make_pair(b,-a);
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

        
        Polinomial u = poly::randomUniformPoly(coef_modulus,poly_modulus,coef_modulus/static_cast<int>(pow(2, degree)));
        Polinomial e0 = poly::randomNormalPoly(coef_modulus,poly_modulus,0,coef_modulus/static_cast<int>(pow(2, degree)) > 1 ? coef_modulus/static_cast<int>(pow(2, degree)) : 2);
        Polinomial e1 = poly::randomNormalPoly(coef_modulus,poly_modulus);
        Polinomial c0_temp1 = poly_eqs::PolyMult_cpu(pk0,u);

        Polinomial c0_temp2 = poly_eqs::PolyMult_cpu(e0,plaintext_modulus);

        Polinomial c0 = poly_eqs::PolyAdd_cpu(poly_eqs::PolyAdd_cpu(c0_temp1,c0_temp2),msg);

        Polinomial c1_temp1 = poly_eqs::PolyMult_cpu(pk1,u);

        Polinomial c1_temp2 = poly_eqs::PolyMult_cpu(e1,plaintext_modulus);

        Polinomial c1 = poly_eqs::PolyAdd_cpu(c1_temp1,c1_temp2);
        return std::make_pair(c0,c1);
    }

    Polinomial decrypt(Polinomial c0, Polinomial c1, Polinomial sk, int64_t plaintext_modulus){
        Polinomial sk_c1 = poly_eqs::PolyMult_cpu(c1,sk);
        Polinomial msg = poly_eqs::PolyAdd_cpu(c0,sk_c1);

        msg.modCenter(plaintext_modulus);
        
        return msg;
    }


    //decrypting multiplied msgs
    Polinomial decrypt_quad(Polinomial c0, Polinomial c1, Polinomial c2, Polinomial sk, int64_t plaintext_modulus){
        Polinomial sk_c1 = poly_eqs::PolyMult_cpu(c1,sk);
        Polinomial sk_c2 = poly_eqs::PolyMult_cpu(c2,sk);
        Polinomial sk_sk_c1 = poly_eqs::PolyMult_cpu(sk_c2,sk);
        Polinomial msg = poly_eqs::PolyAdd_cpu(poly_eqs::PolyAdd_cpu(c0,sk_c1),sk_sk_c1);

        msg.modCenter(plaintext_modulus);    
        return msg;
    }

    int64_t logBase(int64_t value, int base) {
        if (value <= 0) {
            throw std::invalid_argument("Value must be positive.");
        }
        if (base <= 1) {
            throw std::invalid_argument("Base must be greater than 1.");
        }

        int64_t result = 0;
        while (value >= base) {
            value /= base;
            ++result;
        }

        return result;
    }

    GeneralArray<int64_t> int2Base(int value, int base, int& digitCount) {
        // Calculate number of digits required
        digitCount = 0;
        int temp = value;
        while (temp > 0) {
            temp /= base;
            ++digitCount;
        }
        if (digitCount == 0) digitCount = 1; // Handle the case for value = 0

        // Allocate memory for the digits
        GeneralArray<int64_t> digits(digitCount);
        temp = value;

        // Extract digits
        for (int i = 0; i < digitCount; ++i) {
            digits[i] = temp % base;
            temp /= base;
        }

        return digits;
    }

    GeneralArray<Polinomial*> poly2Base(Polinomial poly, int base){
        int n_terms = ceil(logBase(poly.getCoeffModulus(),base));
        int degree = poly.getPolyModSize() - 1;
        if(degree <= 0 && n_terms <= 0){
            printf("Poly2Base: degree or n_terms cannot be 0");
        }
        GeneralArray<GeneralArray<int64_t>> coeffs(degree);

        for (int i = 0; i < degree; ++i) {
            coeffs[i] = GeneralArray<int64_t>(n_terms);
            for (int j = 0; j < n_terms; j++) {
                coeffs[i][j] = 0;
            }
        }


        for (int i = 0; i < degree; ++i){
            int digitCount = 0;
            GeneralArray<int64_t> digits = int2Base(poly[i] % poly.getCoeffModulus(),base,digitCount);
            for (int j = 0; j < digitCount; ++j) {
                coeffs[i][j] = digits[j];
            }
            for (int j = digitCount; j < n_terms; ++j) {
                coeffs[i][j] = 0;
            }
        }

        GeneralArray<Polinomial*> poly_list(n_terms);
        for (size_t i = 0; i < n_terms; i++) {
            poly_list[i] = new Polinomial(degree,poly.getCoeffModulus(),poly.getPolyModulus());
            for (size_t j = 0; j < degree; j++) {
                (*poly_list[i])[j] = coeffs[j][i];
            }
        }

        return poly_list;
    }


    std::pair<Polinomial,Polinomial> Relinearization(Polinomial c0, Polinomial c1, Polinomial c2, GeneralArray<std::pair<Polinomial,Polinomial>*> eks, int base, int64_t coef_modulus, int64_t poly_modulus){
        auto c2_polys = poly2Base(c2,base);

        Polinomial c0_hat = c0;
        Polinomial c1_hat = c1;
        for (size_t i = 0; i < eks.getSize(); i++) {
            c0_hat = poly_eqs::PolyAdd_cpu(c0_hat, poly_eqs::PolyMult_cpu(*c2_polys[i],(*eks[i]).first));
            c1_hat = poly_eqs::PolyAdd_cpu(c1_hat, poly_eqs::PolyMult_cpu(*c2_polys[i],(*eks[i]).second));
        }
        return std::make_pair(c0_hat,c1_hat);
    }


    bool isNoiseSmallEnough(const Polinomial& noise, double threshold) {
        double norm = computeNoiseNorm(noise);
        return norm < threshold;  // Check if the noise norm is below the threshold
    }


}
