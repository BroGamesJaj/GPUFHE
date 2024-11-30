#include <iostream>
#include "headers/GenArray.h"

int main(){
    GenArray heo = GenArray();
    heo.resize(2000);
    for (size_t i = 0; i < heo.getSize(); i++)
    {
        heo[i] = 6534+i;
        std::cout<< heo[i] << std::endl;
    }
    
    
    std::cout<< "heo" << std::endl;
}
