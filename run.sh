docker image inspect gpufhe-base:latest >/dev/null 2>&1 && : || docker build -t gpufhe-base -f Dockerfile.base .

docker build -t gpufhe-build .

if [ $? -ne 0 ]; then
  exit 1
else
  docker run --rm --gpus all gpufhe-build
fi