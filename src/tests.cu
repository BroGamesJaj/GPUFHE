#include "tests.cuh"
#define N 10

namespace tests{
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
            poly_eqs::PolyAdd_gpu<<<block_num,256>>>(d_a, d_b, d_c, N);
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
            poly_eqs::PolySub_gpu<<<block_num,256>>>(d_a, d_b, d_c, N);
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

    void PublicKeyTest(Polinomial pk0, Polinomial pk1, Polinomial sk, Polinomial a, Polinomial e, int64_t plaintext_modulus){
        Polinomial temp1 = poly_eqs::PolyMult_cpu(a,sk);
        Polinomial temp2 = poly_eqs::PolyMult_cpu(e,plaintext_modulus);
        Polinomial reconstructed_pk0 = poly_eqs::PolyAdd_cpu(temp1,temp2);
        reconstructed_pk0.modCenter();

        Polinomial temp3 = poly_eqs::PolyAdd_cpu(pk0, poly_eqs::PolyMult_cpu(pk1,sk));
        if(pk0 == reconstructed_pk0){
            printf("public key 0s are same\n");
        }else{
            printf("public key 0s are NOT same\n");
            pk0.print();
            reconstructed_pk0.print();
        }    
    }


    void cAddTest(int64_t n, int64_t coef_modulus, int64_t plaintext_modulus, GeneralArray<int64_t> poly_modulus, Polinomial sk, std::pair<Polinomial,Polinomial> pk, int64_t batch, int64_t msg_size){
        printf("Összeadás teszt\n");
        Polinomial msg = poly::randomUniformPolyMSG(coef_modulus,poly_modulus, msg_size, batch);
        Polinomial msg2 = poly::randomUniformPolyMSG(coef_modulus,poly_modulus, msg_size, batch);
        printf("Üzenet1: \n");
        msg.print();
        printf("Üzenet2: \n");
        msg2.print();

        auto e_msg = bgvfhe_gpu::asymetricEncryption(pk.first, pk.second,msg,plaintext_modulus,coef_modulus,poly_modulus,n);
        auto e_msg2 = bgvfhe_gpu::asymetricEncryption(pk.first, pk.second,msg2,plaintext_modulus,coef_modulus,poly_modulus,n);
        printf("Titkosított üzenet1:\n");
        e_msg.first.print();
        printf("Titkosított üzenet2:\n");
        e_msg2.first.print();
        printf("CPU összeadás benchmark ...\n");
        double cpu_total_time = 0.0;
        for (int i = 0; i < 20; i++) {
            double start_time = get_time();
            auto result = cypertext_eqs::cAdd_cpu(e_msg, e_msg2);
            double end_time = get_time();
            cpu_total_time += end_time - start_time;
        }
        double cpu_avg_time = cpu_total_time / 20.0;
        auto result = cypertext_eqs::cAdd_cpu(e_msg, e_msg2);
        Polinomial d_msg_msg2 = bgvfhe_gpu::decrypt(result.first,result.second,sk,plaintext_modulus);
        printf("Összeadás eredménye:\n");
        d_msg_msg2.print();
        bool correct = false;
        Polinomial check = poly_eqs::PolyAdd_cpu(msg,msg2);
        if(d_msg_msg2 == check){
            correct = true;
        }
        printf("CPU átlag idő: %f milliseconds\n", cpu_avg_time*1000);
        //printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
        //printf("GPU Device Pointer average time: %f milliseconds\n", gpu2_avg_time*1000);
        //printf("GPU average time: %f milliseconds\n", gpu3_avg_time*1000);
        //printf("Speedup: %fx\n", cpu_avg_time / gpu3_avg_time);
        printf("Az eredmények %s\n", correct ? "megegyeznek" : "nem egyeznek");
        //printf("Results2 are %s\n", correct2 ? "correct" : "incorrect");
        //printf("Results are %s\n", correct3 ? "correct" : "incorrect");ű
        printf("\n\n");
    }

    void cSubTest(int64_t n, int64_t coef_modulus, int64_t plaintext_modulus, GeneralArray<int64_t> poly_modulus, Polinomial sk, std::pair<Polinomial,Polinomial> pk, int64_t batch, int64_t msg_size){
        printf("Kivonás teszt\n");
        Polinomial msg = poly::randomUniformPolyMSG(coef_modulus,poly_modulus, msg_size, batch);
        Polinomial msg2 = poly::randomUniformPolyMSG(coef_modulus,poly_modulus, msg_size, batch);
        printf("Üzenet1: \n");
        msg.print();
        printf("Üzenet2: \n");
        msg2.print();

        auto e_msg = bgvfhe_gpu::asymetricEncryption(pk.first, pk.second,msg,plaintext_modulus,coef_modulus,poly_modulus,n);
        auto e_msg2 = bgvfhe_gpu::asymetricEncryption(pk.first, pk.second,msg2,plaintext_modulus,coef_modulus,poly_modulus,n);
        printf("Titkosított üzenet1:\n");
        e_msg.first.print();
        printf("Titkosított üzenet2:\n");
        e_msg2.first.print();
        printf("CPU kivonás Benchmark ...\n");
        double cpu_total_time = 0.0;
        for (int i = 0; i < 20; i++) {
            double start_time = get_time();
            auto result = cypertext_eqs::cSub_cpu(e_msg, e_msg2);
            double end_time = get_time();
            cpu_total_time += end_time - start_time;
        }
        double cpu_avg_time = cpu_total_time / 20.0;
        auto result = cypertext_eqs::cSub_cpu(e_msg, e_msg2);
        Polinomial d_msg_msg2 = bgvfhe_gpu::decrypt(result.first,result.second,sk,plaintext_modulus);
        printf("Kivonás eredménye:\n");
        d_msg_msg2.print();
        bool correct = false;
        Polinomial check = poly_eqs::PolySub_cpu(msg,msg2);
        if(d_msg_msg2 == check){
            correct = true;
        }
        printf("CPU átlag idő: %f milliseconds\n", cpu_avg_time*1000);
        //printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
        //printf("GPU Device Pointer average time: %f milliseconds\n", gpu2_avg_time*1000);
        //printf("GPU average time: %f milliseconds\n", gpu3_avg_time*1000);
        //printf("Speedup: %fx\n", cpu_avg_time / gpu3_avg_time);
        printf("Az eredmények %s\n", correct ? "megegyeznek" : "nem egyeznek");
        //printf("Results2 are %s\n", correct2 ? "correct" : "incorrect");
        //printf("Results are %s\n", correct3 ? "correct" : "incorrect");
        printf("\n\n");
    }

    void cMultTest(int64_t n, int64_t coef_modulus, int64_t plaintext_modulus, GeneralArray<int64_t> poly_modulus, Polinomial sk, std::pair<Polinomial,Polinomial> pk, int64_t batch, int64_t msg_size){
        printf("Szorzás teszt\n");
        Polinomial msg = poly::randomUniformPolyMSG(coef_modulus,poly_modulus, msg_size,batch);
        Polinomial msg2 = poly::randomUniformPolyMSG(coef_modulus,poly_modulus, msg_size,batch);
        printf("Üzenet1: \n");
        msg.print();
        printf("Üzenet2: \n");
        msg2.print();

        auto e_msg = bgvfhe_gpu::asymetricEncryption(pk.first, pk.second,msg,plaintext_modulus,coef_modulus,poly_modulus,n);
        auto e_msg2 = bgvfhe_gpu::asymetricEncryption(pk.first, pk.second,msg2,plaintext_modulus,coef_modulus,poly_modulus,n);
        printf("Titkosított üzenet1:\n");
        e_msg.first.print();
        printf("Titkosított üzenet2:\n");
        e_msg2.first.print();
        printf("CPU szorzás Benchmark ...\n");
        double cpu_total_time = 0.0;
        auto result = cypertext_eqs::cMult_cpu(e_msg, e_msg2);
        for (int i = 0; i < 20; i++) {
            double start_time = get_time();
            auto result = cypertext_eqs::cMult_cpu(e_msg, e_msg2);
            double end_time = get_time();
            cpu_total_time += end_time - start_time;
        }
        double cpu_avg_time = cpu_total_time / 20.0;
        
        Polinomial d_msg_msg2 = bgvfhe_gpu::decrypt_quad(result.c0,result.c1,result.c2,sk,plaintext_modulus);
        printf("Szorzás eredménye:\n");
        d_msg_msg2.print();
        bool correct = true;
        Polinomial check = poly_eqs::PolyMult_cpu(msg,msg2);
        for (size_t i = 0; i < batch*2-1; i++) {
            if(d_msg_msg2[i] != check[i]){
                correct = false;
            }
        }
        printf("CPU átlag idő: %f milliseconds\n", cpu_avg_time*1000);
        //printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
        //printf("GPU Device Pointer average time: %f milliseconds\n", gpu2_avg_time*1000);
        //printf("GPU average time: %f milliseconds\n", gpu3_avg_time*1000);
        //printf("Speedup: %fx\n", cpu_avg_time / gpu3_avg_time);
        printf("Az eredmények %s\n", correct ? "megegyeznek" : "nem egyeznek");
        //printf("Results2 are %s\n", correct2 ? "correct" : "incorrect");
        //printf("Results are %s\n", correct3 ? "correct" : "incorrect");
        printf("\n\n");
    }

    
}
