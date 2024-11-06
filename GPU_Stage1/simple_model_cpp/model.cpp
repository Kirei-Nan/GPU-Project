#include "model.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>  // for std::max

SimpleModel::SimpleModel() {
    input_size = 512;
    hidden_size = 256;
    output_size = 10;
}

void SimpleModel::load_parameters(
    const std::vector<float> &feature_extractor_0_weight,
    const std::vector<float> &feature_extractor_0_bias,
    const std::vector<float> &feature_extractor_3_weight,
    const std::vector<float> &feature_extractor_3_bias,
    const std::vector<float> &classifier_0_weight,
    const std::vector<float> &classifier_0_bias,
    const std::vector<float> &classifier_3_weight,
    const std::vector<float> &classifier_3_bias) {
    this->feature_extractor_0_weight = feature_extractor_0_weight;
    this->feature_extractor_0_bias = feature_extractor_0_bias;
    this->feature_extractor_3_weight = feature_extractor_3_weight;
    this->feature_extractor_3_bias = feature_extractor_3_bias;
    this->classifier_0_weight = classifier_0_weight;
    this->classifier_0_bias = classifier_0_bias;
    this->classifier_3_weight = classifier_3_weight;
    this->classifier_3_bias = classifier_3_bias;
}

std::vector<float> SimpleModel::forward(const std::vector<float> &input) {
    // Linear layer 1: input * weight + bias
    std::vector<float> hidden1(hidden_size, 0.0f);
    for (int i = 0; i < hidden_size; ++i) {
        float sum = feature_extractor_0_bias[i];
        for (int j = 0; j < input_size; ++j) {
            sum += input[j] * feature_extractor_0_weight[i * input_size + j];
        }
        hidden1[i] = std::max(0.0f, sum);  // ReLU activation
    }

    // Linear layer 2: hidden1 * weight + bias
    std::vector<float> hidden2(hidden_size, 0.0f);
    for (int i = 0; i < hidden_size; ++i) {
        float sum = feature_extractor_3_bias[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden1[j] * feature_extractor_3_weight[i * hidden_size + j];
        }
        hidden2[i] = std::max(0.0f, sum);  // ReLU activation
    }

    // Linear layer 3: hidden2 * weight + bias
    std::vector<float> hidden3(hidden_size, 0.0f);
    for (int i = 0; i < hidden_size; ++i) {
        float sum = classifier_0_bias[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden2[j] * classifier_0_weight[i * hidden_size + j];
        }
        hidden3[i] = std::max(0.0f, sum);  // ReLU activation
    }

    // Output layer: hidden3 * weight + bias
    std::vector<float> output(output_size, 0.0f);
    for (int i = 0; i < output_size; ++i) {
        float sum = classifier_3_bias[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden3[j] * classifier_3_weight[i * hidden_size + j];
        }
        output[i] = sum;
    }

    return output;
}

std::vector<float> load_weights(const std::string &filename) {
    std::vector<float> weights;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + filename);
    }
    std::string line;
    while (std::getline(infile, line)) {
        float value;
        std::istringstream iss(line);
        iss >> value;
        weights.push_back(value);
    }
    infile.close();
    return weights;
}

std::vector<float> load_weights_binary(const std::string &filename, size_t expected_size) {
    std::vector<float> weights(expected_size);
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + filename);
    }
    infile.read(reinterpret_cast<char*>(weights.data()), expected_size * sizeof(float));
    if (!infile) {
        throw std::runtime_error("Error reading weight file: " + filename);
    }
    infile.close();
    return weights;
}