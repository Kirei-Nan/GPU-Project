#include "model.h"
#include <iostream>

int main() {
    try {
        // Select loading method
        bool use_binary = false;  // If true, use binary loading; otherwise use text format loading

        std::vector<float> feature_extractor_0_weight, feature_extractor_0_bias;
        std::vector<float> feature_extractor_3_weight, feature_extractor_3_bias;
        std::vector<float> classifier_0_weight, classifier_0_bias;
        std::vector<float> classifier_3_weight, classifier_3_bias;

        if (use_binary) {
            size_t feature_extractor_0_weight_size = 512 * 256;
            size_t feature_extractor_0_bias_size = 256;
            size_t feature_extractor_3_weight_size = 256 * 256;
            size_t feature_extractor_3_bias_size = 256;
            size_t classifier_0_weight_size = 256 * 256;
            size_t classifier_0_bias_size = 256;
            size_t classifier_3_weight_size = 256 * 10;
            size_t classifier_3_bias_size = 10;

            feature_extractor_0_weight = load_weights_binary("../weights_bin/feature_extractor_0_weight.bin", feature_extractor_0_weight_size);
            feature_extractor_0_bias = load_weights_binary("../weights_bin/feature_extractor_0_bias.bin", feature_extractor_0_bias_size);
            feature_extractor_3_weight = load_weights_binary("../weights_bin/feature_extractor_3_weight.bin", feature_extractor_3_weight_size);
            feature_extractor_3_bias = load_weights_binary("../weights_bin/feature_extractor_3_bias.bin", feature_extractor_3_bias_size);
            classifier_0_weight = load_weights_binary("../weights_bin/classifier_0_weight.bin", classifier_0_weight_size);
            classifier_0_bias = load_weights_binary("../weights_bin/classifier_0_bias.bin", classifier_0_bias_size);
            classifier_3_weight = load_weights_binary("../weights_bin/classifier_3_weight.bin", classifier_3_weight_size);
            classifier_3_bias = load_weights_binary("../weights_bin/classifier_3_bias.bin", classifier_3_bias_size);
        } else {
            feature_extractor_0_weight = load_weights("../weights/feature_extractor_0_weight.txt");
            feature_extractor_0_bias = load_weights("../weights/feature_extractor_0_bias.txt");
            feature_extractor_3_weight = load_weights("../weights/feature_extractor_3_weight.txt");
            feature_extractor_3_bias = load_weights("../weights/feature_extractor_3_bias.txt");
            classifier_0_weight = load_weights("../weights/classifier_0_weight.txt");
            classifier_0_bias = load_weights("../weights/classifier_0_bias.txt");
            classifier_3_weight = load_weights("../weights/classifier_3_weight.txt");
            classifier_3_bias = load_weights("../weights/classifier_3_bias.txt");
        }

        // Initialize the model and load parameters
        SimpleModel model;
        model.load_parameters(feature_extractor_0_weight, feature_extractor_0_bias,
                              feature_extractor_3_weight, feature_extractor_3_bias,
                              classifier_0_weight, classifier_0_bias,
                              classifier_3_weight, classifier_3_bias);

        // Prepare input data (example)
        std::vector<float> input(512, 1.0f);  // Input vector of all ones

        // Forward propagation
        std::vector<float> output = model.forward(input);

        // Output the result
        std::cout << "Model output:" << std::endl;
        for (float val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
