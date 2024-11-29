## DO NOT USE IN PRODUCTION

## Requirements
Docker
NVIDIA GPU/run it remote
CUDA toolkit
C++ interpreter
<!-- to be continued -->
## Running 
Build and run the project using these commands \
bash: \
./run.sh

debug: \
./debug.sh
    cd build
    cuda-gdb GPUFHEApp


debugging using cpu: \
docker build -t gpufhe-base -f Dockerfile.base . \
docker build -t gpufhe-build . \
docker run --rm --cpus all gpufhe-build 
<!-- there maybe issues idk -->
