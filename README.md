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
