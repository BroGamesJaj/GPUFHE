FROM gpufhe-base:latest

RUN mkdir build && cd build && cmake .. && make

CMD ["./build/GPUFHEApp"]