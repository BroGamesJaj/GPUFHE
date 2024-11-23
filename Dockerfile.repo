FROM ghcr.io/rbence01/gpufhe-base:latest

COPY . .

RUN mkdir build && cd build && cmake .. && make

CMD ["./build/GPUFHEApp"]