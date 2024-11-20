// sentiment_analysis.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <cstdlib> // for std::atoi
#include <chrono>

// Include CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace Eigen;

// Define types for row-major matrices and vectors
typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfRowMajor;
typedef Matrix<float, Dynamic, 1> VectorXf;

// Macros for CUDA error checking
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                            \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// Function to read the shape of a parameter from the shape file
vector<int> read_shape(const string& shape_file_path) {
    ifstream shape_file(shape_file_path);
    if (!shape_file.is_open()) {
        cerr << "Error opening shape file: " << shape_file_path << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    getline(shape_file, line);
    shape_file.close();

    vector<int> shape;
    size_t pos = 0;
    while ((pos = line.find(',')) != string::npos) {
        shape.push_back(stoi(line.substr(0, pos)));
        line.erase(0, pos + 1);
    }
    shape.push_back(stoi(line));
    return shape;
}

// Function to load a parameter (weights or biases) from a binary file
template <typename T>
void load_parameter(const string& bin_file_path, const vector<int>& shape, vector<T>& data) {
    ifstream bin_file(bin_file_path, ios::binary);
    if (!bin_file.is_open()) {
        cerr << "Error opening binary file: " << bin_file_path << endl;
        exit(EXIT_FAILURE);
    }

    bin_file.seekg(0, ios::end);
    size_t file_size = bin_file.tellg();
    bin_file.seekg(0, ios::beg);

    size_t num_elements = 1;
    for (int dim : shape) {
        num_elements *= dim;
    }

    if (file_size != num_elements * sizeof(float)) {
        cerr << "File size does not match expected size for: " << bin_file_path << endl;
        exit(EXIT_FAILURE);
    }

    data.resize(num_elements);
    bin_file.read(reinterpret_cast<char*>(data.data()), file_size);
    bin_file.close();
}

// Activation function: ReLU
void relu(MatrixXfRowMajor& x) {
    x = x.cwiseMax(0.0f);
}

// CUDA kernels
__global__ void add_bias_relu(float* x, const float* bias, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx < total) {
        int col = idx % n;
        x[idx] += bias[col];
        if (x[idx] < 0) x[idx] = 0;
    }
}

__global__ void add_bias_relu_vector(float* x, const float* bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += bias[idx];
        if (x[idx] < 0) x[idx] = 0;
    }
}

__global__ void add_bias(float* x, const float* bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += bias[idx];
    }
}

__global__ void set_ones(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = 1.0f;
    }
}

// Custom CUDA kernel for matrix-matrix multiplication
__global__ void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    // Each thread computes one element of the output matrix C
    int row = blockIdx.y * blockDim.y + threadIdx.y; // m
    int col = blockIdx.x * blockDim.x + threadIdx.x; // n
    if (row < m && col < n) {
        float value = 0.0f;
        for (int e = 0; e < k; ++e) {
            value += A[row * k + e] * B[e * n + col];
        }
        C[row * n + col] = value;
    }
}


__global__ void matmul_shared(const float* A, const float* B, float* C, int m, int n, int k) {
    // Shared memory for tiles of A and B
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];

    // Thread and block indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles of size 32x32
    for (int tile = 0; tile < (k + 32 - 1) / 32; ++tile) {
        // Load tiles into shared memory
        if (row < m && (tile * 32 + threadIdx.x) < k) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * k + tile * 32 + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && (tile * 32 + threadIdx.y) < k) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * n + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int e = 0; e < 32; ++e) {
            value += shared_A[threadIdx.y][e] * shared_B[e][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result
    if (row < m && col < n) {
        C[row * n + col] = value;
    }
}



// Custom CUDA kernel for matrix-vector multiplication
__global__ void matvec(const float* A, const float* x, float* y, int m, int n) {
    // Each thread computes one element of the output vector y
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float value = 0.0f;
        for (int e = 0; e < n; ++e) {
            value += A[row * n + e] * x[e];
        }
        y[row] = value;
    }
}


__global__ void matvec_shared(const float* A, const float* x, float* y, int m, int n) {
    // Shared memory for vector x
    __shared__ float shared_x[64]; // Assuming n <= 64 for simplicity

    // Thread index
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Load x into shared memory
    if (threadIdx.x < n) {
        shared_x[threadIdx.x] = x[threadIdx.x];
    }
    __syncthreads();

    // Compute the dot product
    if (row < m) {
        float value = 0.0f;
        for (int e = 0; e < n; ++e) {
            value += A[row * n + e] * shared_x[e];
        }
        y[row] = value;
    }
}




// Kernel to perform mean pooling over tokens (summing and then scaling)
__global__ void mean_pooling(const float* x, float* pooled_features, int tokens, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < features) {
        float sum = 0.0f;
        for (int i = 0; i < tokens; ++i) {
            sum += x[i * features + idx];
        }
        pooled_features[idx] = sum;
    }
}

// Kernel to scale a vector by a scalar
__global__ void scale_vector(float* x, int n, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= scalar;
    }
}


// Model class
class SentimentModel {
public:
    // Constructor
    SentimentModel(const string& weights_dir) {
        // Initialize device pointers to nullptr
        d_fc1_weight = nullptr;
        d_fc1_bias = nullptr;
        d_fc2_weight = nullptr;
        d_fc2_bias = nullptr;
        d_fc3_weight = nullptr;
        d_fc3_bias = nullptr;
        d_fc4_weight = nullptr;
        d_fc4_bias = nullptr;

        load_weights(weights_dir);
    }

    // Destructor
    ~SentimentModel() {
        // Free device memory if allocated
        if (d_fc1_weight) cudaFree(d_fc1_weight);
        if (d_fc1_bias) cudaFree(d_fc1_bias);
        if (d_fc2_weight) cudaFree(d_fc2_weight);
        if (d_fc2_bias) cudaFree(d_fc2_bias);
        if (d_fc3_weight) cudaFree(d_fc3_weight);
        if (d_fc3_bias) cudaFree(d_fc3_bias);
        if (d_fc4_weight) cudaFree(d_fc4_weight);
        if (d_fc4_bias) cudaFree(d_fc4_bias);
    }

    // Function to move model weights to GPU
    void move2gpu() {
        // fc1_weight and fc1_bias (transpose before moving to GPU)
        MatrixXfRowMajor fc1_weight_transposed = fc1_weight.transpose();
        size_t fc1_weight_size = fc1_weight_transposed.rows() * fc1_weight_transposed.cols() * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_fc1_weight, fc1_weight_size));
        CUDA_CHECK(cudaMemcpy(d_fc1_weight, fc1_weight_transposed.data(), fc1_weight_size, cudaMemcpyHostToDevice));

        // Transpose bias (as a row vector)
        MatrixXfRowMajor fc1_bias_transposed = fc1_bias.transpose();
        size_t fc1_bias_size = fc1_bias_transposed.rows() * fc1_bias_transposed.cols() * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_fc1_bias, fc1_bias_size));
        CUDA_CHECK(cudaMemcpy(d_fc1_bias, fc1_bias_transposed.data(), fc1_bias_size, cudaMemcpyHostToDevice));

        // fc2_weight and fc2_bias (transpose before moving to GPU)
        MatrixXfRowMajor fc2_weight_transposed = fc2_weight.transpose();
        size_t fc2_weight_size = fc2_weight_transposed.rows() * fc2_weight_transposed.cols() * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_fc2_weight, fc2_weight_size));
        CUDA_CHECK(cudaMemcpy(d_fc2_weight, fc2_weight_transposed.data(), fc2_weight_size, cudaMemcpyHostToDevice));

        // Transpose bias (as a row vector)
        MatrixXfRowMajor fc2_bias_transposed = fc2_bias.transpose();
        size_t fc2_bias_size = fc2_bias_transposed.rows() * fc2_bias_transposed.cols() * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_fc2_bias, fc2_bias_size));
        CUDA_CHECK(cudaMemcpy(d_fc2_bias, fc2_bias_transposed.data(), fc2_bias_size, cudaMemcpyHostToDevice));

        // fc3_weight and fc3_bias (transpose before moving to GPU)
        MatrixXfRowMajor fc3_weight_transposed = fc3_weight.transpose();
        size_t fc3_weight_size = fc3_weight_transposed.rows() * fc3_weight_transposed.cols() * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_fc3_weight, fc3_weight_size));
        CUDA_CHECK(cudaMemcpy(d_fc3_weight, fc3_weight_transposed.data(), fc3_weight_size, cudaMemcpyHostToDevice));

        // Transpose bias (as a row vector)
        MatrixXfRowMajor fc3_bias_transposed = fc3_bias.transpose();
        size_t fc3_bias_size = fc3_bias_transposed.rows() * fc3_bias_transposed.cols() * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_fc3_bias, fc3_bias_size));
        CUDA_CHECK(cudaMemcpy(d_fc3_bias, fc3_bias_transposed.data(), fc3_bias_size, cudaMemcpyHostToDevice));

        // fc4_weight and fc4_bias (transpose before moving to GPU)
        MatrixXfRowMajor fc4_weight_transposed = fc4_weight.transpose();
        size_t fc4_weight_size = fc4_weight_transposed.rows() * fc4_weight_transposed.cols() * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_fc4_weight, fc4_weight_size));
        CUDA_CHECK(cudaMemcpy(d_fc4_weight, fc4_weight_transposed.data(), fc4_weight_size, cudaMemcpyHostToDevice));

        // Transpose bias (as a row vector)
        MatrixXfRowMajor fc4_bias_transposed = fc4_bias.transpose();
        size_t fc4_bias_size = fc4_bias_transposed.rows() * fc4_bias_transposed.cols() * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_fc4_bias, fc4_bias_size));
        CUDA_CHECK(cudaMemcpy(d_fc4_bias, fc4_bias_transposed.data(), fc4_bias_size, cudaMemcpyHostToDevice));

        cout << "[info] Model weights and biases (transposed) moved to GPU." << endl;
    }

    // CUDA forward pass without cuBLAS
    VectorXf cuda_forward(const MatrixXfRowMajor& input) {
        // Input shape: (256, 300)

        // Copy input to device
        size_t input_size = input.rows() * input.cols() * sizeof(float);
        float* d_input;
        CUDA_CHECK(cudaMalloc((void**)&d_input, input_size));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice));

        // Allocate device memory for intermediate activations
        // x1: (256, 150)
        float* d_x1;
        size_t x1_size = 256 * 150 * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_x1, x1_size));

        // x2: (256, 64)
        float* d_x2;
        size_t x2_size = 256 * 64 * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_x2, x2_size));

        // pooled_features: (64,)
        float* d_pooled_features;
        size_t pooled_features_size = 64 * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_pooled_features, pooled_features_size));

        // x3: (32,)
        float* d_x3;
        size_t x3_size = 32 * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_x3, x3_size));

        // output: (2,)
        float* d_output;
        size_t output_size = 2 * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_output, output_size));

        // Dimensions
        int m, n, k;

        // Layer 1: x1 = input * fc1_weight^T + fc1_bias
        m = 256;
        k = 300;
        n = 150;

        // Define block and grid sizes
        dim3 blockDim(16, 16);
        dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                     (m + blockDim.y - 1) / blockDim.y);

        // Matrix multiplication kernel
        // matmul<<<gridDim, blockDim>>>(d_input, d_fc1_weight, d_x1, m, n, k);
        matmul_shared<<<gridDim, blockDim>>>(d_input, d_fc1_weight, d_x1, m, n, k);
        CUDA_CHECK(cudaGetLastError());

        // Add bias and ReLU
        int threads_per_block = 256;
        int total_elements = m * n;
        int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        add_bias_relu<<<blocks, threads_per_block>>>(d_x1, d_fc1_bias, m, n);
        CUDA_CHECK(cudaGetLastError());

        // Layer 2: x2 = x1 * fc2_weight^T + fc2_bias
        m = 256;
        k = 150;
        n = 64;

        gridDim = dim3((n + blockDim.x - 1) / blockDim.x,
                       (m + blockDim.y - 1) / blockDim.y);

        // matmul<<<gridDim, blockDim>>>(d_x1, d_fc2_weight, d_x2, m, n, k);
        matmul_shared<<<gridDim, blockDim>>>(d_x1, d_fc2_weight, d_x2, m, n, k);
        CUDA_CHECK(cudaGetLastError());

        // Add bias and ReLU
        total_elements = m * n;
        blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        add_bias_relu<<<blocks, threads_per_block>>>(d_x2, d_fc2_bias, m, n);
        CUDA_CHECK(cudaGetLastError());

        // Mean pooling over tokens (axis=0)
        // Initialize pooled_features to zero
        CUDA_CHECK(cudaMemset(d_pooled_features, 0.0f, pooled_features_size));

        // Sum over rows
        int features = n;
        int tokens = m;
        blockDim = dim3(256);
        gridDim = dim3((features + blockDim.x - 1) / blockDim.x);

        // Kernel to sum over rows
        mean_pooling<<<gridDim, blockDim>>>(d_x2, d_pooled_features, tokens, features);
        CUDA_CHECK(cudaGetLastError());

        // Divide by number of tokens to get mean
        scale_vector<<<gridDim, blockDim>>>(d_pooled_features, features, 1.0f / tokens);
        CUDA_CHECK(cudaGetLastError());

        // Layer 3: x3 = fc3_weight * pooled_features + fc3_bias
        m = 32;
        n = 64;

        blockDim = dim3(256);
        gridDim = dim3((m + blockDim.x - 1) / blockDim.x);

        // matvec<<<gridDim, blockDim>>>(d_fc3_weight, d_pooled_features, d_x3, m, n);
        matvec_shared<<<gridDim, blockDim>>>(d_fc3_weight, d_pooled_features, d_x3, m, n);
        CUDA_CHECK(cudaGetLastError());

        // Add bias and ReLU
        threads_per_block = 256;
        blocks = (m + threads_per_block - 1) / threads_per_block;
        add_bias_relu_vector<<<blocks, threads_per_block>>>(d_x3, d_fc3_bias, m);
        CUDA_CHECK(cudaGetLastError());

        // Layer 4: output = fc4_weight * x3 + fc4_bias
        m = 2;
        n = 32;

        gridDim = dim3((m + blockDim.x - 1) / blockDim.x);

        // matvec<<<gridDim, blockDim>>>(d_fc4_weight, d_x3, d_output, m, n);
        matvec_shared<<<gridDim, blockDim>>>(d_fc4_weight, d_x3, d_output, m, n);
        CUDA_CHECK(cudaGetLastError());

        // Add bias
        blocks = (m + threads_per_block - 1) / threads_per_block;
        add_bias<<<blocks, threads_per_block>>>(d_output, d_fc4_bias, m);
        CUDA_CHECK(cudaGetLastError());

        // Copy output to host
        VectorXf output(2);
        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

        // Free device memory for activations
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_x1));
        CUDA_CHECK(cudaFree(d_x2));
        CUDA_CHECK(cudaFree(d_pooled_features));
        CUDA_CHECK(cudaFree(d_x3));
        CUDA_CHECK(cudaFree(d_output));

        return output;
    }

    // Original CPU forward pass
    VectorXf forward(const MatrixXfRowMajor& input) {
        // Input shape: (256, 300)

        // Feature extractor
        // Layer 1: Linear(300, 150) + ReLU
        MatrixXfRowMajor x1 = (input * fc1_weight.transpose()).rowwise() + fc1_bias.transpose();
        relu(x1);

        // Layer 2: Linear(150, 64) + ReLU
        MatrixXfRowMajor x2 = (x1 * fc2_weight.transpose()).rowwise() + fc2_bias.transpose();
        relu(x2);

        // Mean pooling over token features (axis=0)
        VectorXf pooled_features = x2.colwise().mean(); // Shape: (64,)

        // Classifier
        // Layer 3: Linear(64, 32) + ReLU
        VectorXf x3 = (fc3_weight * pooled_features) + fc3_bias;
        x3 = x3.cwiseMax(0.0f);

        // Layer 4: Linear(32, 2)
        VectorXf output = (fc4_weight * x3) + fc4_bias; // Shape: (2,)

        return output;
    }

private:
    // Model parameters
    MatrixXfRowMajor fc1_weight; // Shape: (150, 300)
    VectorXf fc1_bias;           // Shape: (150,)
    MatrixXfRowMajor fc2_weight; // Shape: (64, 150)
    VectorXf fc2_bias;           // Shape: (64,)
    MatrixXfRowMajor fc3_weight; // Shape: (32, 64)
    VectorXf fc3_bias;           // Shape: (32,)
    MatrixXfRowMajor fc4_weight; // Shape: (2, 32)
    VectorXf fc4_bias;           // Shape: (2,)

    // Device pointers for model parameters
    float* d_fc1_weight;
    float* d_fc1_bias;
    float* d_fc2_weight;
    float* d_fc2_bias;
    float* d_fc3_weight;
    float* d_fc3_bias;
    float* d_fc4_weight;
    float* d_fc4_bias;

    // Function to load model weights
    void load_weights(const string& weights_dir) {
        // Layer names mapping
        struct LayerInfo {
            string weight_file;
            string bias_file;
            string weight_shape_file;
            string bias_shape_file;
        };

        vector<LayerInfo> layers = {
                {"feature_extractor_0_weight.bin", "feature_extractor_0_bias.bin",
                        "feature_extractor_0_weight_shape.txt", "feature_extractor_0_bias_shape.txt"},
                {"feature_extractor_3_weight.bin", "feature_extractor_3_bias.bin",
                        "feature_extractor_3_weight_shape.txt", "feature_extractor_3_bias_shape.txt"},
                {"classifier_0_weight.bin", "classifier_0_bias.bin",
                        "classifier_0_weight_shape.txt", "classifier_0_bias_shape.txt"},
                {"classifier_3_weight.bin", "classifier_3_bias.bin",
                        "classifier_3_weight_shape.txt", "classifier_3_bias_shape.txt"}
        };

        // Load each layer's weights and biases
        for (size_t i = 0; i < layers.size(); ++i) {
            // Load weights
            string weight_bin_path = weights_dir + "/" + layers[i].weight_file;
            string weight_shape_path = weights_dir + "/" + layers[i].weight_shape_file;
            vector<int> weight_shape = read_shape(weight_shape_path);
            vector<float> weight_data;
            load_parameter(weight_bin_path, weight_shape, weight_data);

            // Load biases
            string bias_bin_path = weights_dir + "/" + layers[i].bias_file;
            string bias_shape_path = weights_dir + "/" + layers[i].bias_shape_file;
            vector<int> bias_shape = read_shape(bias_shape_path);
            vector<float> bias_data;
            load_parameter(bias_bin_path, bias_shape, bias_data);

            // Assign to model parameters
            if (i == 0) {
                fc1_weight = Map<MatrixXfRowMajor>(weight_data.data(), weight_shape[0], weight_shape[1]);
                fc1_bias = Map<VectorXf>(bias_data.data(), bias_shape[0]);
            } else if (i == 1) {
                fc2_weight = Map<MatrixXfRowMajor>(weight_data.data(), weight_shape[0], weight_shape[1]);
                fc2_bias = Map<VectorXf>(bias_data.data(), bias_shape[0]);
            } else if (i == 2) {
                fc3_weight = Map<MatrixXfRowMajor>(weight_data.data(), weight_shape[0], weight_shape[1]);
                fc3_bias = Map<VectorXf>(bias_data.data(), bias_shape[0]);
            } else if (i == 3) {
                fc4_weight = Map<MatrixXfRowMajor>(weight_data.data(), weight_shape[0], weight_shape[1]);
                fc4_bias = Map<VectorXf>(bias_data.data(), bias_shape[0]);
            }
        }

        cout << "[info] Model weights loaded successfully." << endl;
    }
};

// Function to load test data
void load_test_data(const string& test_data_dir,
                    vector<MatrixXfRowMajor>& inputs,
                    vector<int>& labels) {
    // Load inputs
    string inputs_path = test_data_dir + "/test_inputs.bin";
    ifstream inputs_file(inputs_path, ios::binary);
    if (!inputs_file.is_open()) {
        cerr << "Error opening test inputs file: " << inputs_path << endl;
        exit(EXIT_FAILURE);
    }
    inputs_file.seekg(0, ios::end);
    size_t inputs_file_size = inputs_file.tellg();
    inputs_file.seekg(0, ios::beg);

    size_t sample_size = 256 * 300; // Each sample has shape (256, 300)
    size_t num_samples = inputs_file_size / (sample_size * sizeof(float));

    vector<float> inputs_data(num_samples * sample_size);
    inputs_file.read(reinterpret_cast<char*>(inputs_data.data()), inputs_file_size);
    inputs_file.close();

    // Reshape inputs
    inputs.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        float* sample_data = inputs_data.data() + i * sample_size;
        MatrixXfRowMajor sample = Map<MatrixXfRowMajor>(sample_data, 256, 300);
        inputs.push_back(sample);
    }

    // Load labels
    string labels_path = test_data_dir + "/test_labels.bin";
    ifstream labels_file(labels_path, ios::binary);
    if (!labels_file.is_open()) {
        cerr << "Error opening test labels file: " << labels_path << endl;
        exit(EXIT_FAILURE);
    }
    labels_file.seekg(0, ios::end);
    size_t labels_file_size = labels_file.tellg();
    labels_file.seekg(0, ios::beg);

    size_t num_labels = labels_file_size / sizeof(int64_t);
    if (num_labels != num_samples) {
        cerr << "Number of labels does not match number of samples." << endl;
        exit(EXIT_FAILURE);
    }

    vector<int64_t> labels_data(num_labels);
    labels_file.read(reinterpret_cast<char*>(labels_data.data()), labels_file_size);
    labels_file.close();

    // Convert labels to int
    labels.reserve(num_labels);
    for (size_t i = 0; i < num_labels; ++i) {
        labels.push_back(static_cast<int>(labels_data[i]));
    }

    cout << "[info] Test data loaded successfully. Total samples: " << num_samples << endl;
}

// Main function
int main(int argc, char* argv[]) {
    int num_iterations = 1;

    if (argc < 2) {
        cout << "Using default num_iterations: " << num_iterations << endl;
    } else {
        num_iterations = std::atoi(argv[1]);
        if (num_iterations <= 0) {
            cerr << "Error: num_iterations must be a positive integer." << endl;
            return EXIT_FAILURE;
        }
    }

    // Paths to weights and test data
    string weights_dir = "weights_bin";       // Ensure this directory contains the binary weight files
    string test_data_dir = "test_data_bin";   // Ensure this directory contains test_inputs.bin and test_labels.bin

    // Load the model
    SentimentModel model(weights_dir);

    // Move model weights to GPU
    model.move2gpu();

    // Load test data
    vector<MatrixXfRowMajor> test_inputs;
    vector<int> test_labels;
    load_test_data(test_data_dir, test_inputs, test_labels);

    // Evaluate the model
    int correct = 0;
    int total = test_labels.size();

    total *= num_iterations;

    cout << "[info] Evaluating the model on test data for " << num_iterations << " iterations..." << endl;
    double total_time = 0.;

    for (size_t i = 0; i < test_inputs.size(); ++i) {
        // Mimic multiple times data volume
        for (int iteration = 0; iteration < num_iterations; ++iteration) {
            auto start_time = chrono::high_resolution_clock::now();

            VectorXf output = model.cuda_forward(test_inputs[i]);

            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            total_time += static_cast<double>(duration) / 1e6;

            // Get predicted label
            int predicted_label;
            output.maxCoeff(&predicted_label);
           
        }
    }

    
    cout << "\nTest Results:" << endl;
    cout << "Total time: " << total_time << "s" << endl;
    cout << "Inference per case: " << total_time / total << "s/case" << endl;

    return 0;
}