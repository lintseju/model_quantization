# model_quantization

A model quantization example using [ONNX](https://onnxruntime.ai/).

## Environment Setup

If you are using GPU, please run the following command before `poetry install`

```bash
sed 's/onnxruntime/onnxruntime-gpu/g' pyproject.toml > pyproject_modified.toml
mv -f pyproject_modified.toml pyproject.toml
```

Install the Python environment:

```bash
pip install poetry
poetry install
```

## Run

```bash
poetry run python main.py
```
