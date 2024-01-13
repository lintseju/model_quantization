# model_quantization

A model quantization example using [ONNX](https://onnxruntime.ai/).

## Environment Setup

Make sure [git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) is installed in your computer.

Install the Python environment:

```bash
git lfs install
pip install poetry
poetry install
```

If you are using GPU, please run the following command after `poetry install`

```bash
poetry remove onnxruntime
poetry add onnxruntime-gpu
```

## Run

```bash
poetry run python main.py
```
