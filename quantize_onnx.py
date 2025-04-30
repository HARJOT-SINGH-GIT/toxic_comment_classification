# quantize_onnx.py

import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from transformers import AutoTokenizer
from datasets import load_dataset
import os

# Load the ONNX model
onnx_model_path = "lora-toxic-classifier.onnx"
quantized_model_path = "lora-toxic-classifier-int8.onnx"

# Prepare calibration dataset 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = load_dataset("sms_spam", split="train[:100]")  # small sample for calibration

import numpy as np

class ToxicCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset, tokenizer):
        self.encoded = tokenizer(dataset['sms'], truncation=True, padding=True, return_tensors="np")

        input_ids = self.encoded["input_ids"].astype(np.int64)
        attention_mask = self.encoded["attention_mask"].astype(np.int64)

        # Add batch dimension to each sample
        self.data_iter = iter([
            {
                "input_ids": np.expand_dims(i, axis=0),         # shape (1, seq_len)
                "attention_mask": np.expand_dims(a, axis=0)     # shape (1, seq_len)
            }
            for i, a in zip(input_ids, attention_mask)
        ])

    def get_next(self):
        return next(self.data_iter, None)

calibration_data_reader = ToxicCalibrationDataReader(dataset, tokenizer)

# Quantize to int8
quantize_static(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    calibration_data_reader=calibration_data_reader,
    quant_format=QuantType.QInt8,
)

print(f"Quantized model saved at: {quantized_model_path}")
