INPUT_PATTERN=$1
nvcc -std=c++17 -g -I ./eigen/ -o $INPUT_PATTERN $INPUT_PATTERN.cu
