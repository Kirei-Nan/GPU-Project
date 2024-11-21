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

using namespace std;
using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfRowMajor;
typedef Matrix<float, Dynamic, 1> VectorXf;

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

class SentimentModel {
public:
    // Constructor
    SentimentModel(const string& weights_dir) {
        load_weights(weights_dir);
    }

    VectorXf forward(const MatrixXfRowMajor& input) {
        // Input shape: (256, 300)
        // Debug: Print input shape and sample values
        // cout << "Input shape: (" << input.rows() << ", " << input.cols() << ")" << endl;
        // cout << "Input sample (first 5 values): " << input.row(0).head(5) << endl;

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
        // cout << "After ReLU3 (first 5 values): " << x3.head(5).transpose() << endl;

        // Layer 4: Linear(32, 2)
        VectorXf output = (fc4_weight * x3) + fc4_bias; // Shape: (2,)
        // cout << "Output logits: " << output.transpose() << endl;

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

    void load_weights(const string& weights_dir) {
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
                // Debug: Print shapes
                cout << "fc1_weight shape: (" << fc1_weight.rows() << ", " << fc1_weight.cols() << ")" << endl;
                // cout << "fc1_bias shape: (" << fc1_bias.size() << ")" << endl;
            } else if (i == 1) {
                fc2_weight = Map<MatrixXfRowMajor>(weight_data.data(), weight_shape[0], weight_shape[1]);
                fc2_bias = Map<VectorXf>(bias_data.data(), bias_shape[0]);
                // cout << "fc2_weight shape: (" << fc2_weight.rows() << ", " << fc2_weight.cols() << ")" << endl;
                // cout << "fc2_bias shape: (" << fc2_bias.size() << ")" << endl;
            } else if (i == 2) {
                fc3_weight = Map<MatrixXfRowMajor>(weight_data.data(), weight_shape[0], weight_shape[1]);
                fc3_bias = Map<VectorXf>(bias_data.data(), bias_shape[0]);
                // cout << "fc3_weight shape: (" << fc3_weight.rows() << ", " << fc3_weight.cols() << ")" << endl;
                // cout << "fc3_bias shape: (" << fc3_bias.size() << ")" << endl;
            } else if (i == 3) {
                fc4_weight = Map<MatrixXfRowMajor>(weight_data.data(), weight_shape[0], weight_shape[1]);
                fc4_bias = Map<VectorXf>(bias_data.data(), bias_shape[0]);
                // cout << "fc4_weight shape: (" << fc4_weight.rows() << ", " << fc4_weight.cols() << ")" << endl;
                // cout << "fc4_bias shape: (" << fc4_bias.size() << ")" << endl;
            }
        }

        cout << "[info] Model weights loaded successfully." << endl;
    }
};

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

// Softmax no use
VectorXf softmax(const VectorXf& logits) {
    VectorXf exp_logits = logits.array().exp();
    float sum_exp = exp_logits.sum();
    return exp_logits / sum_exp;
}

int main(int argc, char* argv[]) {

    int num_iterations = 1;

    if (argc < 2) {
        cout << "Using default num_iterations: " << num_iterations << endl;
    } else{
        num_iterations = std::atoi(argv[1]);
        if (num_iterations <= 0) {
            cerr << "Error: num_iterations must be a positive integer." << endl;
            return EXIT_FAILURE;
        }
    }

    string weights_dir = "weights_bin";       // Ensure this directory contains the binary weight files
    string test_data_dir = "test_data_bin";   // Ensure this directory contains test_inputs.bin and test_labels.bin

    SentimentModel model(weights_dir);

    vector<MatrixXfRowMajor> test_inputs;
    vector<int> test_labels;
    load_test_data(test_data_dir, test_inputs, test_labels);

    // Evaluate the model
    int total = test_labels.size();

    total *= num_iterations;

    cout << "[info] Evaluating the model on test data for " << num_iterations << " iterations..." << endl;

    double total_time = 0.;

    for (size_t i = 0; i < test_inputs.size(); ++i) {
        
        // mimic multiple times data volume
        for (int iteration = 0; iteration < num_iterations; ++iteration) {

            auto start_time = chrono::high_resolution_clock::now();

            VectorXf output = model.forward(test_inputs[i]);

            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            total_time += static_cast<double>(duration) / 1e6;

            // Get predicted label
            int predicted_label;
            output.maxCoeff(&predicted_label); // Corrected usage

        }
    }

    cout << "\nTest Results:" << endl;
    cout << "Total time: " << total_time << "s" << endl;
    cout << "Inference per case: " << total_time / total << "s/case" << endl;

    return 0;
}