"""Export a trained resume-fit checkpoint to ONNX for CPU inference, quantize it to int8,
and verify both steps.

The fp32 export is ~255MB -- over GitHub's 100MB hard push limit -- so int8 dynamic
quantization (~67MB) is a required step here, not an optional optimization. Also saves a
standalone tokenizer.json so the Streamlit app can tokenize with the lightweight
`tokenizers` package instead of pulling in full `transformers` + `torch`.

Usage:
    python export_model.py \
        --checkpoint /kaggle/working/runs/resumefit_distilbert_v1/checkpoints/best \
        --output ../../models/resume_fit_distilbert.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

LABELS = ["No Fit", "Potential Fit", "Good Fit"]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.checkpoint)
    model = DistilBertForSequenceClassification.from_pretrained(args.checkpoint)
    model.eval()

    dummy_len = 32
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, dummy_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, dummy_len), dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output.with_suffix(".fp32.onnx")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,  # legacy TorchScript-based exporter; avoids the onnxscript/onnx_ir dependency
    )
    print(f"exported to {fp32_path}")

    with torch.no_grad():
        torch_out = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask).logits.numpy()

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    })[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"max abs diff in logits (torch vs onnx): {max_diff:.6f}")

    torch_prob = softmax(torch_out)
    onnx_prob = softmax(onnx_out)
    max_prob_diff = np.abs(torch_prob - onnx_prob).max()
    print(f"max abs diff in probability (torch vs onnx): {max_prob_diff:.6f}")
    assert max_prob_diff < 0.01, "ONNX export does not match PyTorch output — do not trust this artifact"
    print("fp32 export verified OK")

    quantize_dynamic(str(fp32_path), str(args.output), weight_type=QuantType.QInt8)
    fp32_size = fp32_path.stat().st_size / 1024 / 1024
    quant_size = args.output.stat().st_size / 1024 / 1024
    fp32_path.unlink()
    print(f"quantized {fp32_size:.1f}MB -> {quant_size:.1f}MB: {args.output}")

    quant_sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    quant_out = quant_sess.run(["logits"], {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    })[0]
    quant_prob_diff = np.abs(torch_prob - softmax(quant_out)).max()
    print(f"max abs diff in probability (torch fp32 vs quantized): {quant_prob_diff:.4f}")
    # Loose tolerance -- quantization is lossy by design (expect a few points of accuracy
    # loss, verified separately via evaluate.py against real data), this just catches a
    # broken/garbage quantized export, not fine-grained accuracy regression.
    assert quant_prob_diff < 0.5, "quantized model diverges too much from fp32 -- do not trust this artifact"
    print("quantized export verified OK")

    # Standalone tokenizer.json (loadable via the `tokenizers` package, no `transformers` needed).
    tokenizer_dst = args.output.parent / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tokenizer_dst))
    print(f"saved tokenizer to {tokenizer_dst}")

    labels_dst = args.output.parent / "labels.json"
    with open(labels_dst, "w") as f:
        json.dump(LABELS, f, indent=2)
    print(f"saved labels to {labels_dst}")


if __name__ == "__main__":
    main()
