nvcc -O3 -std=c++17 -arch=native cells_cuda_network_mp4.cu -o cells_cuda_network_mp4 \
  $(pkg-config --cflags --libs opencv4)
