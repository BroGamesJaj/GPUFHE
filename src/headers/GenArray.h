#pragma once
#include <iostream>
#include <stdexcept>

namespace general_array
{
    class GenArray
    {
    private:
        int64_t *array;
        size_t size;

    public:
        GenArray(size_t initialSize = 0, bool fill = true);

        GenArray(size_t initialSize, int64_t* array);

        ~GenArray();

        GenArray(const GenArray &other);

        GenArray(GenArray &&other) noexcept;

        GenArray &operator=(const GenArray &other);

        GenArray operator=(GenArray &&other) noexcept;

        void setArray(int64_t* value, bool del = true);

        void resize(size_t new_size);

        void pop_back();

        int64_t *getArray() const;

        size_t getSize() const;

        int64_t &operator[](size_t index);

        const int64_t &operator[](size_t index) const;

        int64_t *begin() const;

        int64_t *end() const;

        int64_t back();

        void Out();

        void PutLast(int64_t number);
    };
}

using general_array::GenArray;