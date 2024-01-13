# model_quantization

A model quantization example using [ONNX](https://onnxruntime.ai/).

For more details, please read my blog in [中文](https://writings.jigfopsda.com/zh/posts/2024/lightweighting_models_using_onnx/) or in [English](https://writings.jigfopsda.com/en/posts/2024/lightweighting_models_using_onnx/).

## Environment Setup

Make sure [git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) is installed in your computer.

Install the Python environment:

```bash
git lfs install
git submodule update --init
pip install poetry
poetry install --no-root
```

If you are using GPU, please run the following command after `poetry install --no-root`

```bash
poetry remove onnxruntime
poetry add onnxruntime-gpu
```

## Run

```bash
poetry run python main.py
```

## Experiment Results on CPU

Experimenting with one of the DistilBERT models fine-tuned on the IMDB dataset from HuggingFace, available [here](https://huggingface.co/lvwerra/distilbert-imdb).

The results running on a MacBook Air M1 CPU and Windows 10 WSL with an i5-8400 CPU are provided below (results may vary on different platforms):

|                       | Model Size | Inference Time per Instance | Accuracy |
|:---------------------:|:----------:|:---------------------------:|:--------:|
| PyTorch Model (MAC)   | 256MB      | 71.1ms                      | 93.8%    |
| ONNX Model(MAC)       | 256MB      | 113.5ms                     | 93.8%    |
| ONNX 8-bit Model(MAC) | 64MB       | 87.7ms                      | 93.75%   |
| PyTorch Model (Win)   | 256MB      | 78.6ms                      | 93.8%    |
| ONNX Model(Win)       | 256MB      | 85.1ms                      | 93.8%    |
| ONNX 8-bit Model(Win) | 64MB       | 61.1ms                      | 93.85%   |
