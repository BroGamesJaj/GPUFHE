## Requirements
Docker
NVIDIA GPU/run it remote
CUDA toolkit
C++ interpreter
<!-- to be continued -->
## Running
In CMakeLists.txt Change the variable of CUDA_ARCHITECTURES from "80" to your gpu

Build and run the project using these commands
bash:
docker build -t gpufhe-build .
docker run --rm --gpus all gpufhe-build

<!-- there maybe issues idk -->