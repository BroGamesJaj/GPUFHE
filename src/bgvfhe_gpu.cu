#include <stdio.h>

__global__ void add(long long int* a,long long int* b,long long int* c){
    int i = threadIdx.x + blockIdx.y * blockDim.x;
    c[i] = a[i] + b[i];
}

__managed__ long long int vector_a[1073741824], vector_b[1073741824], vector_c[1073741824];

int main(){
    for (int i = 0; i < 1073741824; i++){
        vector_a[i] = i;
        vector_b[i] = 1073741824 - i;
    }

    add<<<1,256>>>(vector_a, vector_b, vector_c);

    cudaDeviceSynchronize();

    
    long long int res = 0;
    for (int i = 0; i < 1073741824; i++){
        res += vector_c[i];
    }
    printf("sum result: % d",res);
}