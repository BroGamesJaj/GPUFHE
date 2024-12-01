#include <iostream>
#include "headers/GenArray.h"
#include "headers/RFHE.h"

int main(){
    RFHE sys(256,10);
    sys.genPK();
    sys.genPuk();

    GenArray** PuK = sys.getPuK();

    int64_t topSecret = 69;
    std::cout << topSecret << std::endl << std::endl;

    GenArray message = sys.EncInt(topSecret);
    message.Out();


    GenArray* encMessage = new GenArray[2];
    sys.Encrypt(message, PuK, encMessage);
    encMessage[0].Out();
    encMessage[1].Out();

    std::cout << "private:";
    sys.getPK().Out();

    GenArray decMessage = sys.Decrypt(encMessage[0], encMessage[1]);
    decMessage.Out();

    int64_t notSecret = sys.DecInt(decMessage);
    std::cout << notSecret << std::endl << std::endl;


}
