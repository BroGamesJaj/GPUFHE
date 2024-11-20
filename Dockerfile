FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

WORKDIR /app

RUN apt-get update && \
apt-get install -y \
build-essential \
cmake \
g++ \
gcc \
cuda-toolkit-12-6 \
wget \
curl \
git

COPY . .

RUN mkdir build
RUN cd build && cmake .. && make

CMD ["./build/GPUFHEApp"]