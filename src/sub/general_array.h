#pragma once

namespace general_array {
    template <typename T>
    class GeneralArray{

    private:
        T* array;
        size_t size;

    public:

        GeneralArray(size_t initialSize = 0): size(initialSize), array(nullptr) {
            if(size > 0){
                array = new T[size];
            }
        }

        void resize(size_t new_size){
            T* tempArray = new T[new_size];
            size_t copy_size = std::min(size, new_size);
            for (size_t i = 0; i < copy_size; ++i) {
                tempArray[i] = array[i];
            }
            delete[] array;
            array = tempArray;
            size = new_size;
        }

        size_t get_size() const {
            return size;
        }

        T& operator[](size_t index) {
            return array[index];
        }

        const T& operator[](size_t index) const {
            return array[index];
        }
    };
}