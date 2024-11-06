#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>

class SimpleModel {
public:
    SimpleModel();
    void load_parameters(
        const std::vector<float> &feature_extractor_0_weight,
        const std::vector<float> &feature_extractor_0_bias,
        const std::vector<float> &feature_extractor_3_weight,
        const std::vector<float> &feature_extractor_3_bias,
        const std::vector<float> &classifier_0_weight,
        const std::vector<float> &classifier_0_bias,
        const std::vector<float> &classifier_3_weight,
        const std::vector<float> &classifier_3_bias);
    std::vector<float> forward(const std::vector<float> &input);

private:
    // Model parameters
    std::vector<float> feature_extractor_0_weight;
    std::vector<float> feature_extractor_0_bias;
    std::vector<float> feature_extractor_3_weight;
    std::vector<float> feature_extractor_3_bias;
    std::vector<float> classifier_0_weight;
    std::vector<float> classifier_0_bias;
    std::vector<float> classifier_3_weight;
    std::vector<float> classifier_3_bias;

    // Model hyperparameters
    int input_size;
    int hidden_size;
    int output_size;
};

std::vector<float> load_weights(const std::string &filename);
std::vector<float> load_weights_binary(const std::string &filename, size_t expected_size);

#endif