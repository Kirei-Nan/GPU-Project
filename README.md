# Accelerating the inference speed of a DNN model using CUDA

Team Number: 14

Team Member: Shilong Dong, Zhenghan Nan, Yifu Deng

## Data and Model Weight Download

First, you need to download the binary format of our test data and pretrained model weight from: [Google Drive Link](https://drive.google.com/drive/folders/1ZqgGORml7wR3m2YVZy-kyvNJmgTr7RT_?usp=sharing), and place the downloaded folder `single_test_sample/`, `test_data_bin/`, `weights/`, and `weights_bin/` under the root dir of this repo.

Make sure the structure of this repo is:

```bash
.
├── Stage1_Pytorch_impl.ipynb
├── Stage2_CPP_seq_impl.cpp
├── Stage3_CUDA_naive_impl.cu
├── Stage3_CUDA_shared_mem_impl.cu
├── Stage3_CUDA_shared_mem_stream_impl.cu
├── eigen/
├── single_test_sample/
├── test_data_bin/
├── weights/
├── weights_bin/
├── README.md
├── script_compile_cuda.sh
└── script_compile_seq.sh

```

## File Description

- `Stage1_Pytorch_impl.ipynb`: The Pytorch training and inferencing notebook.
- `Stage2_CPP_seq_impl.cpp`: The C++ inference implementation.
- `Stage3_CUDA_naive_impl.cu`: The CUDA naive implementation.
- `Stage3_CUDA_shared_mem_impl.cu`: The CUDA implementation optimized by shared memory.
- `Stage3_CUDA_shared_mem_stream_impl.cu`: The CUDA implementation optimized by shared memory and stream technique.

## Running

We conduct the `Stage1` of this project on M1 Max Chip Mac, so don't run the stage1 notebook on CIMS machines.

For the rest stages (stage2 and stage3), we run them on CIMS cuda5 server. 

- The example compile and running commad for stage2 is:

```bash
bash script_compile_seq.sh Stage2_CPP_seq_impl
./Stage2_CPP_seq_impl 1
```

- The example compile and running commad for stage3 is:

```bash
bash script_compile_cuda.sh Stage3_CUDA_naive_impl
./Stage3_CUDA_naive_impl 1
```

NOTE1: The arg `1`  in an example cmd (like `./Stage3_CUDA_naive_impl 1` ) means we run the whole test dataset 1 iteration. This arg simulates how many times the amount of data we are testing on the test dataset for our model. You can set this to `10`, `100` or even larger to assume the model is running on a new dataset with 10x, 100x or larger times of the current test dataset.

NOTE2: Ignore all the warnings during compling, they don't affect the results.
