#include "headers/GenArray.h"
#include <string>

GenArray::GenArray(size_t initialSize, bool fill)
    : size(initialSize), array(nullptr)
{
    if (size > 0)
    {
        array = new int64_t[size];
        if (fill)
            std::fill_n(array, size, int64_t());
    }
}

GenArray::GenArray(size_t initialSize, int64_t* array) : size(initialSize), array(array) {}

GenArray::~GenArray(){
    delete[] array;
}

GenArray::GenArray(const GenArray &other) : size(other.size), array(new int64_t[other.size])
{
    std::copy(other.begin(), other.end(), array);
}

GenArray::GenArray(GenArray &&other) noexcept
    : size(other.size), array(other.array)
{
    other.array = nullptr;
    other.size = 0;
}

GenArray& GenArray::operator=(const GenArray &other){
    if (this != &other)
            {
                int64_t *newArray = new int64_t[other.size];
                std::copy(other.begin(), other.end(), newArray);

                if( array != nullptr ){
                    delete[] array;
                }
                array = newArray;
                size = other.size;
            }
            return *this;
}

GenArray GenArray::operator=(GenArray &&other) noexcept {
    if (this != &other)
    {
        if( array != nullptr ){
            delete[] array;
        }
        array = other.array;
        size = other.size;
        
        other.array = nullptr;
        other.size = 0;
    }

    return *this;
}

void GenArray::setArray(int64_t* value, bool del) {
    if (del) delete[] array;
    array = value;
}

void GenArray::resize(size_t new_size)
  {
    int64_t* tempArray = new int64_t[new_size];
    size_t copy_size = std::min(size, new_size);
    for (size_t i = 0; i < copy_size; ++i){
        tempArray[i] = array[i];
    }
    if (new_size > size){
        for (size_t i = size; i < new_size; ++i){
            tempArray[i] = int64_t();
        }
    }
    if( array != nullptr ){
        delete[] array;
    }
    array = tempArray;
    size = new_size;
}

void GenArray::pop_back() {
    if (size == 0) {
        throw std::out_of_range("Cannot pop from an empty array");
    }
    resize(size - 1);
} 

int64_t* GenArray::getArray() const {
    return array;
}

size_t GenArray::getSize() const
{
    return size;
}

int64_t& GenArray::operator[](size_t index)
{
    return array[index];
}

const int64_t& GenArray::operator[](size_t index) const
{
    return array[index];
}

int64_t* GenArray::begin() const
{
    return array;
}

int64_t* GenArray::end() const
{
    return array + size;
}

int64_t GenArray::back() {
    if (size == 0) {
        throw std::out_of_range("No elements in the array");
    }
    return array[size - 1];
}

void GenArray::Out(){
    std::string out;
    for (size_t i = 0; i < size; i++)
    {
        out += std::to_string(array[i])+"x^"+std::to_string(i);
        if(i < size-1) out += " + ";
    }
    std::cout<< out << std::endl << std::endl;
}

void GenArray::PutLast(int64_t number){
    resize(size+1);
    array[size-1] = number;
}


