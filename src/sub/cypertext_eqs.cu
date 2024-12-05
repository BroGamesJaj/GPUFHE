#include "cypertext_eqs.h"

namespace cypertext_eqs{
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

    struct result cMult_cpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2){
        Polinomial temp1 = poly_eqs::PolyMult_cpu(e_msg1.first,e_msg2.first);
        Polinomial temp2 = poly_eqs::PolyAdd_cpu(poly_eqs::PolyMult_cpu(e_msg1.first,e_msg2.second),poly_eqs::PolyMult_cpu(e_msg1.second,e_msg2.first));
        temp2.modCenter();
        Polinomial temp3 = poly_eqs::PolyMult_cpu(e_msg1.second,e_msg2.second);
        return {temp1,temp2,temp3};
    }

    std::pair<Polinomial, Polinomial> cAdd_gpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2) {
        size_t n = e_msg1.first.getSize();
        int blockSize = 256;
        int numBlocks = (n + blockSize - 2) / blockSize;

        int64_t *msg1_1, *msg1_2, *msg2_1, *msg2_2, *result_1, *result_2;
        cudaMalloc(&msg1_1, n * sizeof(int64_t));
        cudaMalloc(&msg1_2, n * sizeof(int64_t));
        cudaMalloc(&msg2_1, n * sizeof(int64_t));
        cudaMalloc(&msg2_2, n * sizeof(int64_t));
        cudaMalloc(&result_1, n * sizeof(int64_t));
        cudaMalloc(&result_2, n * sizeof(int64_t));
        cudaMemcpy(msg1_1, e_msg1.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(msg1_2, e_msg1.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(msg2_1, e_msg1.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(msg2_2, e_msg1.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);

        int64_t modulo = e_msg1.first.getCoeffModulus();
        
        poly_eqs::PolyAdd_gpu<<<numBlocks, blockSize>>>(msg1_1, msg2_1, result_1, n);
        poly_eqs::ModCenter_gpu<<<numBlocks, blockSize>>>(result_1, modulo, n);
        poly_eqs::PolyAdd_gpu<<<numBlocks, blockSize>>>(msg1_2, msg2_2, result_2, n);
        poly_eqs::ModCenter_gpu<<<numBlocks, blockSize>>>(result_1, modulo, n);
        int64_t *result_1_h = (int64_t*)malloc(n * sizeof(int64_t));
        int64_t *result_2_h = (int64_t*)malloc(n * sizeof(int64_t));
        cudaDeviceSynchronize();
        cudaMemcpy(result_1_h, result_1, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_2_h, result_2, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaFree(msg1_1);
        cudaFree(msg1_2);
        cudaFree(msg2_1);
        cudaFree(msg2_2);
        cudaFree(result_1);
        cudaFree(result_2);
        Polinomial result1(n, result_1_h);
        Polinomial result2(n, result_2_h);
        result1.setCoeffModulus(modulo);
        result2.setCoeffModulus(modulo);
        result1.setPolyModulus(e_msg1.first.getPolyModulus());
        result2.setPolyModulus(e_msg1.first.getPolyModulus());
        return {result1, result2};
    }

    std::pair<Polinomial, Polinomial> cSub_gpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2){
        size_t n = e_msg1.first.getSize();
        int blockSize = 256;
        int numBlocks = (n + blockSize - 2) / blockSize;

        int64_t *msg1_1, *msg1_2, *msg2_1, *msg2_2, *result_1, *result_2;
        cudaMalloc(&msg1_1, n * sizeof(int64_t));
        cudaMalloc(&msg1_2, n * sizeof(int64_t));
        cudaMalloc(&msg2_1, n * sizeof(int64_t));
        cudaMalloc(&msg2_2, n * sizeof(int64_t));
        cudaMalloc(&result_1, n * sizeof(int64_t));
        cudaMalloc(&result_2, n * sizeof(int64_t));
        cudaMemcpy(msg1_1, e_msg1.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(msg1_2, e_msg1.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(msg2_1, e_msg1.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(msg2_2, e_msg1.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        
        int64_t modulo = e_msg1.first.getCoeffModulus();
        poly_eqs::PolySub_gpu<<<numBlocks, blockSize>>>(msg1_1, msg2_1, result_1, n);
        poly_eqs::ModCenter_gpu<<<numBlocks, blockSize>>>(result_1, modulo, n);
        poly_eqs::PolySub_gpu<<<numBlocks, blockSize>>>(msg1_2, msg2_2, result_2, n);
        poly_eqs::ModCenter_gpu<<<numBlocks, blockSize>>>(result_2, modulo, n);

        int64_t *result_1_h = (int64_t*)malloc(n * sizeof(int64_t));
        int64_t *result_2_h = (int64_t*)malloc(n * sizeof(int64_t));
        cudaDeviceSynchronize();
        cudaMemcpy(result_1_h, result_1, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_2_h, result_2, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaFree(msg1_1);
        cudaFree(msg1_2);
        cudaFree(msg2_1);
        cudaFree(msg2_2);
        cudaFree(result_1);
        cudaFree(result_2);
        Polinomial result1(n, result_1_h);
        Polinomial result2(n, result_2_h);
        result1.setCoeffModulus(modulo);
        result2.setCoeffModulus(modulo);
        result1.setPolyModulus(e_msg1.first.getPolyModulus());
        result2.setPolyModulus(e_msg1.first.getPolyModulus());
        return {result1, result2};  
    }

    struct result cMult_gpu(std::pair<Polinomial,Polinomial> e_msg1, std::pair<Polinomial,Polinomial> e_msg2){
        size_t n = e_msg1.first.getSize();
        int blockSize = 256;
        int numBlocks = (n + blockSize - 2) / blockSize;
        int multNumBlocks = ((2 * n) + blockSize - 2) / blockSize;
        int64_t modulo = e_msg1.first.getCoeffModulus();

        int64_t *msg1_1, *msg1_2, *msg2_1, *msg2_2, *result_1, *result_2, *result_2_reduced, *result_3, *temp_1, *temp_2, *modulus;
        cudaMalloc(&msg1_1, n * sizeof(int64_t));
        cudaMalloc(&msg1_2, n * sizeof(int64_t));
        cudaMalloc(&msg2_1, n * sizeof(int64_t));
        cudaMalloc(&msg2_2, n * sizeof(int64_t));
        cudaMalloc(&result_1, (2*n-1) * sizeof(int64_t));
        cudaMalloc(&result_2, (2*n-1) * sizeof(int64_t));
        cudaMalloc(&result_3, (2*n-1) * sizeof(int64_t));
        cudaMalloc(&temp_1, (2*n-1) * sizeof(int64_t));
        cudaMalloc(&temp_2, (2*n-1) * sizeof(int64_t));
        cudaMalloc(&modulus, (n+1) * sizeof(int64_t));
        cudaMemcpy(msg1_1, e_msg1.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(msg1_2, e_msg1.second.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(msg2_1, e_msg2.first.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(msg2_2, e_msg2.second.getCoeffPointer(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(modulus, e_msg1.first.getPolyModulus().getArray(), (n+1) * sizeof(int64_t), cudaMemcpyHostToDevice);

        auto res = poly_eqs::PolyDiv_cpu(e_msg1.first, e_msg1.second);
        printf("n: %d\n", n);
        printf("size of mult n: %d\n", res.second.getSize());
        printf("div CPU: ");
        for (int i = 0; i < n; i++) printf("%d ", res.second[i]);
        printf("\n");

        int64_t* a;
        cudaMalloc(&a, n * sizeof(int64_t));
        int64_t* asd = (int64_t*)malloc(n * sizeof(int64_t));
        cudaMemcpy(asd, msg1_1, n * sizeof(int64_t), cudaMemcpyDeviceToHost);

        bool match = true;
        for (int i = 0; i < n; i++) {
            if (asd[i] != e_msg1.first[i]) match = false;
        }
        printf("match: %d \n", match);

        poly_eqs::PolyDiv_gpu<<<numBlocks, blockSize>>>(msg1_1, a, msg1_2, n, n);
        //poly_eqs::PolyMult_gpu<<<multNumBlocks, blockSize>>>(msg1_1, msg1_2, msg2_1, 2*n-1);
        //poly_eqs::ReducePoly_gpu(msg2_1, modulus, modulo, 2*n-1, n+1);
        cudaDeviceSynchronize();
        int64_t* modcenter = (int64_t*)malloc(n * sizeof(int64_t));
        cudaMemcpy(modcenter, msg2_1, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        printf("div GPU: ");
        for (int i = 0; i < n; i++) printf("%d ", modcenter[i]);
        printf("\n");

        printf("\n");
        printf("\n");
        printf("\n");

        
        poly_eqs::PolyMult_gpu<<<multNumBlocks, blockSize>>>(msg1_1, msg2_1, result_1, 2*n-1);
        poly_eqs::PolyMult_gpu<<<multNumBlocks, blockSize>>>(msg1_1, msg2_2, temp_1, 2*n-1);
        poly_eqs::PolyMult_gpu<<<multNumBlocks, blockSize>>>(msg1_2, msg2_1, temp_2, 2*n-1);
        cudaDeviceSynchronize();
        
        poly_eqs::ReducePoly_gpu(result_1, modulus, modulo, 2*n-1, n+1);
        poly_eqs::ReducePoly_gpu(temp_1, modulus, modulo, 2*n-1, n+1);
        poly_eqs::ReducePoly_gpu(temp_2, modulus, modulo, 2*n-1, n+1);
        cudaDeviceSynchronize();

        poly_eqs::PolyAdd_gpu<<<multNumBlocks, blockSize>>>(temp_1, temp_2, result_2, n);
        cudaDeviceSynchronize();
        poly_eqs::ReducePoly_gpu(result_2, modulus, modulo, 2*n-1, n+1);
        cudaDeviceSynchronize();
        poly_eqs::ModCenter_gpu<<<numBlocks, blockSize>>>(result_2, modulo, n);
        poly_eqs::PolyMult_gpu<<<multNumBlocks, blockSize>>>(msg1_2, msg2_2, result_3, 2*n-1);
        cudaDeviceSynchronize();
        poly_eqs::ReducePoly_gpu(result_3, modulus, modulo, 2*n-1, n+1);

        int64_t *result_1_h = (int64_t*)malloc(n * sizeof(int64_t));
        int64_t *result_2_h = (int64_t*)malloc(n * sizeof(int64_t));
        int64_t *result_3_h = (int64_t*)malloc(n * sizeof(int64_t));
        cudaDeviceSynchronize();

        cudaMemcpy(result_1_h, result_1, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_2_h, result_2, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_3_h, result_3, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaFree(msg1_1);
        cudaFree(msg1_2);
        cudaFree(msg2_1);
        cudaFree(msg2_2);
        cudaFree(result_1);
        cudaFree(result_2);
        cudaFree(result_3);
        cudaFree(temp_1);
        cudaFree(temp_2);

        Polinomial result1(n, result_1_h);
        Polinomial result2(n, result_2_h);
        Polinomial result3(n, result_3_h);

        result1.setCoeffModulus(modulo);
        result2.setCoeffModulus(modulo);
        result3.setCoeffModulus(modulo);

        result1.setPolyModulus(e_msg1.first.getPolyModulus());
        result2.setPolyModulus(e_msg1.first.getPolyModulus());
        result3.setPolyModulus(e_msg1.first.getPolyModulus());

        return {result1, result2, result3};
    }
}