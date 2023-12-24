import logging
import os
from time import perf_counter

import numpy as np
import onnxruntime
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
    CalibrationDataReader
)
import torch
from torch.utils.data import DataLoader
import torchtext
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"


class CalibrationLoader(CalibrationDataReader):
    def __init__(self, tokenizer, data):
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)

        self.batches = []
        batch_size=32
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            encoded = [tokenizer.encode(d) for d in batch]
            input_ids = np.array([e.ids for e in encoded])
            attention_mask = np.array([e.attention_mask for e in encoded])
            onnx_input = {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "token_type_ids": np.array([np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64),
            }
            self.batches.append(onnx_input)

    def get_next(self):
        print(f'next called - {len(self.batches)}')
        if len(self.batches) == 0:
            return None
        else:
            return self.batches.pop()


def inference_torch(model, batch_dict, labels):
    start = perf_counter()
    labels = labels.to(device)
    with torch.no_grad():
        output = model(**batch_dict)
    if device == "cuda":
        output = output.cpu()
        labels = labels.cpu()
    inference_time = perf_counter() - start
    return output, inference_time


def inference_onnx(model, batch_dict):
    start = perf_counter()
    output = model.run(None, batch_dict)
    inference_time = perf_counter() - start
    return output[0], inference_time


def eval_model(dataset, dataset_length, model, tokenizer, batch_size=32, model_type="pt") -> float:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    corrects = 0
    total = 0
    total_time = 0
    for labels, texts in tqdm(dataloader, total=(dataset_length - 1) // batch_size + 1):
        batch_dict = tokenizer(texts, return_tensors="pt", max_length=256, truncation=True, padding=True)
        if model_type == "pt":
            output, inference_time = inference_torch(model, batch_dict, labels)
            # Label of the IMDB dataset are 1 and 2.
            corrects += (torch.argmax(output.logits, dim=1) + 1 == labels).sum().item()
        else:
            feed_dict = {"input_ids": batch_dict["input_ids"].numpy(),
                         "attention_mask": batch_dict["attention_mask"].numpy()}
            output, inference_time = inference_onnx(model, feed_dict)
            # Label of the IMDB dataset are 1 and 2.
            corrects += (np.argmax(output, axis=1) + 1 == labels.numpy()).sum()
        total += len(labels)
        total_time += inference_time
    logging.info(f"Evaluation time: {total_time:.4f}s")
    return corrects / total


def main():
    logging.info("Loading pretrained models: lvwerra/distilbert-imdb")
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    torch.save(model.state_dict(), "models/distilbert-imdb.pt")
    dataset = torchtext.datasets.IMDB(split='test')
    dataset_length = 25000
    model = model.to(device)
    model.eval()
    logging.info("Evaluating PyTorch model running time and accuracy.")
    logging.info("Accuracy %.4f", eval_model(dataset, dataset_length, model, tokenizer))

    logging.info("Convert PyTorch model to ONNX.")
    example_inputs = tokenizer("query: this is a test sentence", return_tensors="pt")
    if device == "cuda":
        model = model.to("cpu")
    torch.onnx.export(
        model,
        tuple((example_inputs['input_ids'], example_inputs['attention_mask'])),
        "models/distilbert-imdb.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "max_seq_len"},
            "attention_mask": {0: "batch_size", 1: "max_seq_len"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
    )

    logging.info("Evaluating ONNX model running time and accuracy.")
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession("models/distilbert-imdb.onnx", sess_options=sess_options, providers=[provider])
    logging.info("Accuracy %.4f", eval_model(dataset, dataset_length, model, tokenizer, model_type="onnx"))

    logging.info("Evaluating quantized ONNX model running time and accuracy.")
    quantize_dynamic(
        model_input="models/distilbert-imdb.onnx",
        model_output="models/distilbert-imdb.int8.onnx",
        weight_type=QuantType.QInt8,
        extra_options=dict(
            EnableSubgraph=True
        ),
    )
    model = onnxruntime.InferenceSession("models/distilbert-imdb.int8.onnx", sess_options=sess_options, providers=[provider])
    logging.info("Accuracy %.4f", eval_model(dataset, dataset_length, model, tokenizer, model_type="onnx"))


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    main()
