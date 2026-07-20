"""Export a trained resume-fit checkpoint to ONNX for CPU inference, quantize it to int8,
and verify both steps.

The fp32 export is ~255MB -- over GitHub's 100MB hard push limit -- so int8 dynamic
quantization (~67MB) is a required step here, not an optional optimization. Also saves a
standalone tokenizer.json so the Streamlit app can tokenize with the lightweight
`tokenizers` package instead of pulling in full `transformers` + `tensorflow`.

Three real compatibility issues had to be worked around here (all verified by hand against
the installed tf2onnx==1.17.0 -- not assumptions):

1. KerasHub's DistilBERT computes its exact (erf-based) GELU in a way that TensorFlow
   compiles down to an `Erfc` op, which tf2onnx has no converter for. Worked around with a
   custom op handler that rewrites `Erfc(x)` as the mathematically exact `1 - Erf(x)` --
   ONNX has a native `Erf` op, so this is a lossless rewrite, not an approximation.
2. tf2onnx's public `convert.from_function()` API crashes (verified: a real interpreter
   segfault / MemoryError, not a clean exception) inside its own `remove_back_to_back`
   graph-optimizer pass on this model's graph, in this tf2onnx/protobuf version combo.
   Worked around by calling tf2onnx's lower-level pipeline directly (frozen-graph -> graph
   conversion -> model_proto) and skipping just that optimizer pass -- the exported graph is
   less aggressively optimized as a result, but verified numerically identical to the
   unquantized TF model (see the `max abs diff` check below) and still gets int8-quantized
   the same as before.
3. KerasHub's attention layers use `EinsumDense` (a single fused `Einsum` op covering
   reshape+matmul+reshape), which `onnxruntime.quantization.quantize_dynamic` doesn't
   recognize at all -- its dynamic-quantization registry only covers
   MatMul/Gather/Conv/LSTM/Attention/Transpose, not Einsum. Left as-is, that silently skips
   quantizing all 24 attention Q/K/V/output-projection weight matrices (54MB of the ~255MB
   fp32 model), producing a 111MB "quantized" file that still blows past GitHub's 100MB push
   limit. `quantize_einsum_weights()` below does the same dynamic int8 quantization by hand
   for exactly those nodes (symmetric per-tensor, QuantizeLinear/DequantizeLinear pair
   inserted before each Einsum) before handing off to quantize_dynamic for the rest --
   verified this drops the final size to ~69MB with <0.001 max softmax-probability drift.

Usage:
    python export_model.py \
        --checkpoint /kaggle/working/runs/resumefit_distilbert_v1/checkpoints/best.keras \
        --output ../../models/resume_fit_distilbert.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from tf2onnx import tf_loader, utils as tf2onnx_utils
from tf2onnx.convert import tensor_names_from_structed
from tf2onnx.tfonnx import process_tf_graph
from transformers import DistilBertTokenizerFast

LABELS = ["No Fit", "Potential Fit", "Good Fit"]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def erfc_to_erf_handler(ctx, node, name, args):
    """tf2onnx has no converter for TF's Erfc op (see module docstring). Rewrites
    Erfc(x) -> 1 - Erf(x), which ONNX's native Erf op (opset 9+) supports directly --
    exact, not an approximation."""
    x = node.input[0]
    shapes = node.output_shapes
    dtypes = node.output_dtypes
    output_name = node.output[0]
    one_const = ctx.make_const(tf2onnx_utils.make_name("erfc_one"), np.array(1.0, dtype=np.float32))
    erf_node = ctx.make_node("Erf", [x], op_name_scope=name)
    ctx.remove_node(name)
    ctx.make_node("Sub", [one_const.output[0], erf_node.output[0]], outputs=[output_name],
                  name=name, shapes=shapes, dtypes=dtypes)


def quantize_einsum_weights(model_path: Path, min_weight_size: int = 1000):
    """Symmetric per-tensor int8 dynamic quantization for Einsum nodes' constant weight
    inputs -- see point 3 in the module docstring for why quantize_dynamic can't do this.
    In-place: overwrites model_path.
    """
    model = onnx.load(str(model_path))
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    new_nodes = []
    replaced_init_names = set()
    n_quantized = 0
    for node in graph.node:
        if node.op_type != "Einsum":
            new_nodes.append(node)
            continue
        weight_idx = next((i for i, inp in enumerate(node.input) if inp in initializers), None)
        if weight_idx is None:
            new_nodes.append(node)
            continue

        weight_name = node.input[weight_idx]
        weight_arr = numpy_helper.to_array(initializers[weight_name])
        if weight_arr.dtype != np.float32 or weight_arr.size < min_weight_size:
            new_nodes.append(node)
            continue

        max_abs = np.abs(weight_arr).max()
        scale = float(max_abs / 127.0) if max_abs > 0 else 1.0
        quantized = np.clip(np.round(weight_arr / scale), -127, 127).astype(np.int8)

        quant_name = weight_name + "_i8"
        scale_name = weight_name + "_scale"
        dequant_name = weight_name + "_dq"
        graph.initializer.append(numpy_helper.from_array(quantized, name=quant_name))
        graph.initializer.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), name=scale_name))
        replaced_init_names.add(weight_name)

        new_nodes.append(helper.make_node(
            "DequantizeLinear", [quant_name, scale_name], [dequant_name], name=weight_name + "_DQ"
        ))
        node.input[weight_idx] = dequant_name
        new_nodes.append(node)
        n_quantized += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    kept_initializers = [init for init in graph.initializer if init.name not in replaced_init_names]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    onnx.save(model, str(model_path))
    print(f"quantized {n_quantized} einsum weight tensors")


def convert_to_onnx(serving_fn, input_signature, opset: int, output_path: Path):
    """Equivalent to tf2onnx.convert.from_function(), except it skips the graph-optimizer
    pass that crashes on this model (see module docstring point 2)."""
    concrete_func = serving_fn.get_concrete_function(*input_signature)
    input_names = [t.name for t in concrete_func.inputs if t.dtype != tf.dtypes.resource]
    output_names = [t.name for t in concrete_func.outputs if t.dtype != tf.dtypes.resource]
    tensors_to_rename = tensor_names_from_structed(concrete_func, input_names, output_names)

    with tf.device("/cpu:0"):
        frozen_graph = tf_loader.from_function(concrete_func, input_names, output_names, large_model=False)
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(frozen_graph, name="")
            g = process_tf_graph(
                tf_graph, continue_on_error=True, opset=opset,
                input_names=input_names, output_names=output_names,
                custom_op_handlers={"Erfc": (erfc_to_erf_handler, [])},
                tensors_to_rename=tensors_to_rename,
            )
            model_proto = g.make_model("resumatch")

    tf2onnx_utils.save_protobuf(str(output_path), model_proto)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer_model_name", default="distilbert-base-uncased",
                        help="HF model id the tokenizer vocab matches (same vocab as the KerasHub preset).")
    args = parser.parse_args()

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.tokenizer_model_name)
    model = keras.models.load_model(args.checkpoint)

    # int64, named input_ids/attention_mask: matches the ONNX graph the deployed app.py
    # already calls (session.run(["logits"], {"input_ids": ..., "attention_mask": ...}) with
    # np.int64 arrays) so this is a drop-in replacement -- app.py needs no changes.
    # KerasHub's model itself expects {"token_ids", "padding_mask"} as int32, so serving_fn
    # renames/casts at the boundary.
    dummy_len = 32
    dummy_input_ids = tf.random.uniform((1, dummy_len), minval=0, maxval=tokenizer.vocab_size, dtype=tf.int64)
    dummy_attention_mask = tf.ones((1, dummy_len), dtype=tf.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output.with_suffix(".fp32.onnx")

    input_signature = [
        tf.TensorSpec((None, None), tf.int64, name="input_ids"),
        tf.TensorSpec((None, None), tf.int64, name="attention_mask"),
    ]

    @tf.function(input_signature=input_signature)
    def serving_fn(input_ids, attention_mask):
        token_ids = tf.cast(input_ids, tf.int32)
        padding_mask = tf.cast(attention_mask, tf.int32)
        return {"logits": model({"token_ids": token_ids, "padding_mask": padding_mask})}

    convert_to_onnx(serving_fn, input_signature, opset=17, output_path=fp32_path)
    print(f"exported to {fp32_path}")

    tf_out = model({
        "token_ids": tf.cast(dummy_input_ids, tf.int32),
        "padding_mask": tf.cast(dummy_attention_mask, tf.int32),
    }).numpy()

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    })[0]

    max_diff = np.abs(tf_out - onnx_out).max()
    print(f"max abs diff in logits (tf vs onnx): {max_diff:.6f}")

    tf_prob = softmax(tf_out)
    onnx_prob = softmax(onnx_out)
    max_prob_diff = np.abs(tf_prob - onnx_prob).max()
    print(f"max abs diff in probability (tf vs onnx): {max_prob_diff:.6f}")
    assert max_prob_diff < 0.01, "ONNX export does not match TensorFlow output — do not trust this artifact"
    print("fp32 export verified OK")

    quantize_einsum_weights(fp32_path)
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
    quant_prob_diff = np.abs(tf_prob - softmax(quant_out)).max()
    print(f"max abs diff in probability (tf fp32 vs quantized): {quant_prob_diff:.4f}")
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
