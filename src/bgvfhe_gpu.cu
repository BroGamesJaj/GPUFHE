#include <stdio.h>

__global__ void add(int* a, int* b, int* c){
    int i = threadIdx.x + blockIdx.y * blockDim.x;
    c[i] = a[i] + b[i];
}

__managed__ int vector_a[16384], vector_b[16384], vector_c[16384];

int main(){
    for (int i = 0; i < 16384; i++){
        vector_a[i] = i;
        vector_b[i] = 16384 - i;
    }

    add<<<1,256>>>(vector_a, vector_b, vector_c);

    cudaDeviceSynchronize();

    
    int res = 0;
    for (int i = 0; i < 16384; i++){
        res += vector_c[i];
    }
    printf("sum result: % d",res);
}