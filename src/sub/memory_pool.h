#pragma once

#include <iostream>

namespace memory_pool{

    class MemoryPool;
    typedef std::shared_ptr<MemoryPool> MemoryPoolHandle;
    class MemoryPool{
        private:
            static MemoryPoolHandle global_pool;
    };
}