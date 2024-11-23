## Requirements
Docker
NVIDIA GPU/run it remote
CUDA toolkit
C++ interpreter
<!-- to be continued -->
## Running 
Build and run the project using these commands \
bash: \
docker build -t gpufhe-base -f Dockerfile.base . \
docker build -t gpufhe-build . \
docker run --rm --gpus all gpufhe-build 


debugging using cpu: \
docker build -t gpufhe-base -f Dockerfile.base . \
docker build -t gpufhe-build . \
docker run --rm --cpus all gpufhe-build 
<!-- there maybe issues idk -->
