#include "bgvfhe_gpu.cuh"
#define N 10

void init_poly(int64_t *array, int n) {
    std::random_device rd;                     // Seed for randomness
    std::mt19937 gen(rd());                    // Mersenne Twister generator
    std::uniform_int_distribution<size_t> dis(1, 10);
    for (size_t i = 0; i < n; ++i) {
        array[i] = dis(gen); // Generate random number and assign to array
    }
}

double get_time() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

void AddTest(){
    printf("test for polynomial addition\n");
    size_t size1 = N * sizeof(int64_t);
    size_t size_out = N * sizeof(int64_t);
    Polinomial array(N);
    Polinomial array2(N);
    Polinomial array3(N);
    Polinomial array_gpu(N);
    int64_t *d_a, *d_b, *d_c;

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        array3 = poly_eqs::PolyAdd_cpu(array,array2);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;
    
    printf("\n");
    cudaMalloc(&d_a, size1);
    cudaMalloc(&d_b, size1);
    cudaMalloc(&d_c, size_out);
    cudaMemset(d_c, 0, size_out);
    cudaMemcpy(d_a, array.getCoeffPointer(), size1, cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, array2.getCoeffPointer(), size1, cudaMemcpyHostToDevice );

    int block_num = (N + 256 - 1) / 256;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        poly_eqs::PolyAdd_gpu<<<block_num,256>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;
    cudaMemcpy(array_gpu.getCoeffPointer(), d_c, size_out, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < array_gpu.getSize(); i++) {
        if(array_gpu[i] - array3[i] != 0){
            correct = false;
            break;
        }
    }
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("\n");
}

void SubTest(){
    printf("Test for Polynomial substration\n");
    size_t size1 = N * sizeof(int64_t);
    size_t size_out = N * sizeof(int64_t);
    Polinomial array(N);
    Polinomial array2(N);
    Polinomial array3(N);
    Polinomial array_gpu(N);
    int64_t *d_a, *d_b, *d_c;

    printf("Benchmarking CPU implementation...\n\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        array3 = poly_eqs::PolySub_cpu(array,array2);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    cudaMalloc(&d_a, size1);
    cudaMalloc(&d_b, size1);
    cudaMalloc(&d_c, size_out);
    cudaMemset(d_c, 0, size_out);
    cudaMemcpy(d_a, array.getCoeffPointer(), size1, cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, array2.getCoeffPointer(), size1, cudaMemcpyHostToDevice );

    int block_num = (N + 256 - 1) / 256;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        poly_eqs::PolySub_gpu<<<block_num,256>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;
    cudaMemcpy(array_gpu.getCoeffPointer(), d_c, size_out, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < array_gpu.getSize(); i++) {
        if(array_gpu[i] - array3[i] != 0){
            correct = false;
            break;
        }
    }
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("\n");
}
 
void MultTest(){
    printf("test for polynomial multiplication\n");
    size_t size1 = N * sizeof(int64_t);
    size_t size_out = (2 * N - 1) * sizeof(int64_t);
    Polinomial array(N);
    Polinomial array2(N);
    Polinomial array3((2 * N -1));
    Polinomial array_gpu((2 * N -1));
    int64_t *d_a, *d_b, *d_c;


    init_poly(array.getCoeffPointer(), array.getSize()); 
    init_poly(array2.getCoeffPointer(), array2.getSize()); 

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        array3 = poly_eqs::PolyMult_cpu(array,array2);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;
    
    printf("\n");
    cudaMalloc(&d_a, size1);
    cudaMalloc(&d_b, size1);
    cudaMalloc(&d_c, size_out);
    cudaMemset(d_c, 0, size_out);
    cudaMemcpy(d_a, array.getCoeffPointer(), size1, cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, array2.getCoeffPointer(), size1, cudaMemcpyHostToDevice );

    int block_num = (2 * N + 256 - 1) / 256;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        poly_eqs::PolyMult_gpu<<<block_num,256>>>(d_a, d_b, d_c, array.getSize());
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;
    cudaMemcpy(array_gpu.getCoeffPointer(), d_c, size_out, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < array_gpu.getSize(); i++) {
        if(array_gpu[i] - array3[i] != 0){
            correct = false;
            break;
        }
    }
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("\n");
}

void DivTest() {
    printf("Test for Polynomial division\n");
    Polinomial dividend(N);
    Polinomial divisor(N);

    init_poly(dividend.getCoeffPointer(), dividend.getSize());
    init_poly(divisor.getCoeffPointer(), divisor.getSize());

    std::pair<Polinomial, Polinomial> res = poly_eqs::PolyDiv_cpu(dividend, divisor);

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        res = poly_eqs::PolyDiv_cpu(dividend, divisor);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;


    printf("Benchmarking GPU implementation...\n");
    /*std::pair<Polinomial, Polinomial> res_gpu = poly_eqs::PolyDiv_gpu(dividend, divisor);
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        res_gpu = poly_eqs::PolyDiv_gpu(dividend, divisor);
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("Benchmarking Device Pointer GPU implementation...\n");*/

    //int64_t *remainder1 = (int64_t*)malloc(dividend.getSize() * sizeof(int64_t));
    int64_t *remainder_d;
    int64_t *divisor_d;
    int64_t* quotient = (int64_t*)malloc(sizeof(int64_t) * dividend.getSize() - divisor.getSize() + 1);
    cudaMalloc(&remainder_d, sizeof(int64_t) * dividend.getSize());
    cudaMalloc(&divisor_d, sizeof(int64_t) * divisor.getSize());
    cudaMemcpy(remainder_d, dividend.getCoeffPointer(), sizeof(int64_t) * dividend.getSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(divisor_d, divisor.getCoeffPointer(), sizeof(int64_t) * divisor.getSize(), cudaMemcpyHostToDevice);
    /*poly_eqs::PolyDivW_gpu(remainder_d, quotient, divisor_d, dividend.getSize(), divisor.getSize());
    double gpu2_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(remainder_d, dividend.getCoeffPointer(), sizeof(int64_t) * dividend.getSize(), cudaMemcpyHostToDevice);
        cudaMemcpy(divisor_d, divisor.getCoeffPointer(), sizeof(int64_t) * divisor.getSize(), cudaMemcpyHostToDevice);
        double start_time = get_time();
        poly_eqs::PolyDivW_gpu(remainder_d, quotient, divisor_d, dividend.getSize(), divisor.getSize());
        double end_time = get_time();
        gpu2_total_time += end_time - start_time;
    }
    double gpu2_avg_time = gpu2_total_time / 20.0;
    cudaMemcpy(remainder1, remainder_d, dividend.getSize(), cudaMemcpyDeviceToHost);*/
    
    int64_t *quotient_d;
    cudaMalloc(&quotient_d, sizeof(int64_t) * (dividend.getSize() - divisor.getSize() + 1));
    int64_t *remainder2 = (int64_t*)malloc(dividend.getSize() * sizeof(int64_t));

    int blockSize = 256;
    int numBlocks = (divisor.getSize() + blockSize - 2) / blockSize;

    poly_eqs::PolyDiv_gpu<<<numBlocks, blockSize>>>(remainder_d, quotient_d, divisor_d, dividend.getSize(), divisor.getSize());
    double gpu3_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(remainder_d, dividend.getCoeffPointer(), sizeof(int64_t) * dividend.getSize(), cudaMemcpyHostToDevice);
        cudaMemcpy(divisor_d, divisor.getCoeffPointer(), sizeof(int64_t) * divisor.getSize(), cudaMemcpyHostToDevice);
        double start_time = get_time();
        poly_eqs::PolyDiv_gpu<<<numBlocks, blockSize>>>(remainder_d, quotient_d, divisor_d, dividend.getSize(), divisor.getSize());
        double end_time = get_time();
        gpu3_total_time += end_time - start_time;
    }
    double gpu3_avg_time = gpu3_total_time / 20.0;

    cudaMemcpy(remainder2, remainder_d, dividend.getSize() * sizeof(int64_t), cudaMemcpyDeviceToHost);

    /*printf("\nCpu:\nQuotient: ");
    for (size_t i = 0; i < res.first.getSize(); i++) {
        printf("%d^%d ", res.first[i], i);
    }
    
    printf("\nRemainder: ");
    for (size_t i = 0; i < res.second.getSize(); i++) {
        printf("%d^%d ", res.second[i], i);
    }

    printf("\nGpu1:\nQuotient: ");
    for (size_t i = 0; i < res_gpu.first.getSize(); i++) {
        printf("%d^%d ", res_gpu.first[i], i);
    }
    
    printf("\nRemainder: ");
    for (size_t i = 0; i < res_gpu.second.getSize(); i++) {
        printf("%d^%d ", res_gpu.second[i], i);
    }

    bool correct = true;
    for (int i = 0; i < res_gpu.first.getSize(); i++) {
        if(res_gpu.first[i] - res.first[i] != 0 || res_gpu.second[i] - res.second[i] != 0) {
            correct = false;
            break;
        }
    }

   
    printf("\nGpu2:\nQuotient: ");
    for (size_t i = 0; i < dividend.getSize() - divisor.getSize() + 1; i++) {
        printf("%d^%d ",quotient[i], i);
    }
    printf("\nRemainder: ");
    for (size_t i = 0; i < dividend.getSize(); i++) {
        printf("%d^%d ", remainder1[i], i);
    }

    bool correct2 = true;
    for (int i = 0; i < dividend.getSize() - divisor.getSize() + 1; i++) {
        if(quotient[i] - res.first[i] != 0 || remainder1[i] - res.second[i] != 0){
            correct2 = false;
            break;
        }
    }*/

    cudaMemcpy(quotient, quotient_d, (dividend.getSize() - divisor.getSize() + 1), cudaMemcpyDeviceToHost);

    /*printf("\nGpu3:\nQuotient: ");
    for (size_t i = 0; i < dividend.getSize() - divisor.getSize() + 1; i++) {
        printf("%d^%d ", quotient[i], i);
    }
    printf("\nRemainder: ");
    for (size_t i = 0; i < dividend.getSize(); i++) {
        printf("%d^%d ", remainder2[i], i);
    }*/

    bool correct3 = true;
    for (int i = 0; i < dividend.getSize() - divisor.getSize() + 1; i++) {
        if(quotient[i] - res.first[i] != 0 || remainder2[i] - res.second[i] != 0){
            correct3 = false;
            break;
        }
    }
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    //printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    //printf("GPU Device Pointer average time: %f milliseconds\n", gpu2_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu3_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu3_avg_time);
    //printf("Results are %s\n", correct ? "correct" : "incorrect");
    //printf("Results2 are %s\n", correct2 ? "correct" : "incorrect");
    printf("Results are %s\n", correct3 ? "correct" : "incorrect");

    cudaFree(remainder_d);
    cudaFree(divisor_d);
    cudaFree(quotient_d);
    delete [] quotient;
    //delete [] remainder1;
    delete [] remainder2;
}

Polinomial GeneratePrivateKey(int64_t coeff_modulus, GeneralArray<int64_t> poly_modulus){
    if(coeff_modulus != 0 && poly_modulus.getSize() != 0){
        Polinomial randomPoly = poly::randomTernaryPoly(coeff_modulus, poly_modulus);
    
        printf("private key sk:\n");
        randomPoly.print();
        return randomPoly;
    }else{
        throw std::runtime_error("coefficient or poly_modulus is not set");
    }
}

void PublicKeyTest(Polinomial pk0, Polinomial pk1, Polinomial sk, Polinomial a, Polinomial e, int64_t plaintext_modulus){
    Polinomial temp1 = poly_eqs::PolyMult_cpu(a,sk);
    //printf("temp 1 after mult a * sk\n");
    //temp1.print();
    Polinomial temp2 = poly_eqs::PolyMult_cpu(e,plaintext_modulus);
    //printf("temp 2 after mult e * plaintext_modulus\n");
    //temp2.print();
    Polinomial reconstructed_pk0 = poly_eqs::PolyAdd_cpu(temp1,e);
    reconstructed_pk0.polyMod();
    //printf("b after adding temp1 + temp2\n");
    //reconstructed_pk0.print();

    Polinomial temp3 = poly_eqs::PolyAdd_cpu(pk0, poly_eqs::PolyMult_cpu(pk1,sk));
    printf("temp3 should be equal to e: ");
    temp3.print();
    printf("e: ");
    e.print();
    if(pk0 == reconstructed_pk0){
        printf("public key 0s are same\n");
    }else{
        printf("public key 0s are NOT same\n");
        pk0.print();
        reconstructed_pk0.print();
    }

    
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

std::pair<Polinomial,Polinomial> GeneratePublicKey(Polinomial& sk, int64_t coeff_modulus, GeneralArray<int64_t>& poly_modulus, int64_t plaintext_modulus){
    Polinomial e = poly::randomNormalPoly(coeff_modulus,poly_modulus);
    //printf("e noise: ");
    //computeNoiseNorm(e);
    //e.print();
    Polinomial a = poly::randomTernaryPoly(coeff_modulus,poly_modulus);
    //printf("a noise: ");
    //computeNoiseNorm(a);
    //a.print();
    //printf("calc start\n");
    //printf("e\n");
    //e.print();
    //printf("a\n");
    //a.print();
    //printf("sk\n");
    //sk.print();
    //sk.print();
    Polinomial temp1 = poly_eqs::PolyMult_cpu(a,sk);
    //printf("temp1 noise: ");
    //computeNoiseNorm(temp1);
    //temp1.print();
    temp1.reducePolynomial();
    //printf("temp 1 after mult a * sk\n");
    //temp1.print();

    Polinomial temp2 = poly_eqs::PolyMult_cpu(e,plaintext_modulus);
    //printf("temp2 noise: ");
    //computeNoiseNorm(temp2);
    //temp2.print();
    temp2.reducePolynomial();
    //printf("temp 2 after mult e * plaintext_modulus\n");
    //temp2.print();

    Polinomial b = poly_eqs::PolyAdd_cpu(temp1,temp2);
    computeNoiseNorm(b);
    b.modCenter();
    //printf("b/pk0 after adding temp1 + temp2\n");
    //b.print();

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
    c0_temp1.modCenter();
    Polinomial c0_temp2 = poly_eqs::PolyMult_cpu(e0,plaintext_modulus);
    c0_temp2.modCenter();
    Polinomial c0 = poly_eqs::PolyAdd_cpu(poly_eqs::PolyAdd_cpu(c0_temp1,c0_temp2),msg);
    //printf("c0 noise: ");
    //computeNoiseNorm(c0);
    c0.reducePolynomial();
    Polinomial c1_temp1 = poly_eqs::PolyMult_cpu(pk1,u);
    c1_temp1.modCenter();
    Polinomial c1_temp2 = poly_eqs::PolyMult_cpu(e1,plaintext_modulus);
    c1_temp2.modCenter();
    Polinomial c1 = poly_eqs::PolyAdd_cpu(c1_temp1,c1_temp2);
    //printf("c1 noise: ");
    //computeNoiseNorm(c1);
    c1.reducePolynomial();
    //printf("c0\n");
    //c0.print();
    //c1.print();
    return std::make_pair(c0,c1);
}

Polinomial decrypt(Polinomial c0, Polinomial c1, Polinomial sk, int64_t plaintext_modulus){
    Polinomial sk_c1 = poly_eqs::PolyMult_cpu(c1,sk);
    //printf("sk_c1 noise: ");
    //computeNoiseNorm(sk_c1);
    sk_c1.reducePolynomial();
    Polinomial msg = poly_eqs::PolySub_cpu(c0,sk_c1);
    //computeNoiseNorm(msg);
    msg.reducePolynomial();
    msg.polyMod(plaintext_modulus);
    
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
    int64_t max_degree = 100; // amount of numbers stored
    for (size_t i = 0; i < 5; i++) {}
    GeneralArray<int64_t> poly_modulus = poly::initPolyModulus(n); 

    
    Polinomial sk = GeneratePrivateKey(coef_modulus, poly_modulus);
    auto result = GeneratePublicKey(sk, coef_modulus, poly_modulus, plaintext_modulus);
    printf("PK generator ended\n");
    //printf("%d",result);
    Polinomial pk0 = result.first;
    Polinomial pk1 = result.second;

    Polinomial msg = poly::randomUniformPolyMSG(coef_modulus, poly_modulus, plaintext_modulus/4,max_degree > n ? n : max_degree);
    Polinomial msg2 = poly::randomUniformPolyMSG(coef_modulus, poly_modulus, plaintext_modulus/4,max_degree > n ? n : max_degree);
    printf("MSG:\n");
    msg.print();
    auto e_msg = asymetricEncryption(pk0,pk1,msg,plaintext_modulus,coef_modulus,poly_modulus,n);
    auto e_msg2 = asymetricEncryption(pk0,pk1,msg,plaintext_modulus,coef_modulus,poly_modulus,n);

    Polinomial decrypted_msg = decrypt(e_msg.first, e_msg.second,sk,plaintext_modulus);
    printf("decrypted MSG:\n");
    decrypted_msg.print();
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