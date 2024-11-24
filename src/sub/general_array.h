#pragma once
#include <iostream>

namespace general_array
{
    template <typename T>
    class GeneralArray
    {

    private:
        T *array;
        size_t size;

    public:
        GeneralArray<T>(size_t initialSize = 0, bool fill = true)
            : size(initialSize), array(nullptr)
        {
            if (size > 0)
            {
                array = new T[size];
                if (fill)
                    std::fill_n(array, size, T());
            }
        }

        ~GeneralArray<T>()
        {
            delete[] array;
        }

        GeneralArray<T>(const GeneralArray<T> &other)
            : size(other.size), array(new T[other.size])
        {
            std::copy(other.begin(), other.end(), array);
        }

        GeneralArray<T>(GeneralArray<T> &&other) noexcept
            : size(other.size), array(other.array)
        {
            other.array = nullptr;
            other.size = 0;
        }

        GeneralArray<T> &operator=(const GeneralArray<T> &other)
        {
            if (this != &other)
            {
                T *newArray = new T[other.size];
                std::copy(other.begin(), other.end(), newArray);

                delete[] array;
                array = newArray;
                size = other.size;
            }
            return *this;
        }

        GeneralArray<T> &operator=(GeneralArray<T> &&other) noexcept
        {
            if (this != &other)
            {
                delete[] array;

                array = other.array;
                size = other.size;

                other.array = nullptr;
                other.size = 0;
            }

            return *this;
        }

        void resize(size_t new_size)
        {
            T *tempArray = new T[new_size];
            size_t copy_size = std::min(size, new_size);
            for (size_t i = 0; i < copy_size; ++i)
            {
                tempArray[i] = array[i];
            }
            delete[] array;
            array = tempArray;
            size = new_size;
        }

        T* getArray() {
            return array;
        }

        size_t getSize() const
        {
            return size;
        }

        T &operator[](size_t index)
        {
            return array[index];
        }

        const T &operator[](size_t index) const
        {
            return array[index];
        }

        T *begin() const
        {
            return array;
        }

        T *end() const
        {
            return array + size;
        }
    };
}

using general_array::GeneralArray;