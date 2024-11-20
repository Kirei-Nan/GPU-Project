INPUT_PATTERN=$1
g++ -std=c++17 -g $INPUT_PATTERN.cpp -o $INPUT_PATTERN -I ./eigen/