docker build -t gpufhe-build .

if [ $? -ne 0 ]; then
  exit 1
else
  docker run --rm --gpus all gpufhe-build
fi