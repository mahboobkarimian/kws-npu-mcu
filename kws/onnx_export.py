"""Build the int8 QDQ ONNX graph for the AMD Phoenix NPU (Ryzen AI 1.8 compiler) from the integer parameters.

The graph is written by hand from export_int8()'s dict, so it holds EXACTLY the integers the MCU header holds:

  mfcc[1,1,49,10] -> Q/DQ -> Conv 10x4/2 -> Relu -> Q/DQ -> 4 x (Conv dw 3x3 -> Relu -> Q/DQ -> Conv 1x1 -> Relu -> Q/DQ)
                  -> GlobalAveragePool -> Q/DQ -> Conv 1x1 (classifier) -> Q/DQ -> logits[1,C,1,1]

Rules from the Phoenix op map (FINDINGS.md sec. 11) and DFN3-NPU (dfnpu_quant.py):
  - every Conv has an explicit `strides` attribute (the compile fails otherwise)
  - no Q/DQ between a Conv and its Relu (the DPU fuses them)
  - weights: int8 + DequantizeLinear; biases: int32 + DequantizeLinear at the int8-bias scale
  - the host feeds the input already on the int8 grid, so the input QuantizeLinear changes nothing
SPDX-License-Identifier: MIT
"""
from typing import Dict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import config as C
from .quant import layer_tag


def build_qdq(P: Dict, num_classes: int, path: str, opset: int = 17, upto: int = None) -> onnx.ModelProto:
    """upto: debugging only -- stop after layer index `upto` and output that layer's int8 tensor."""
    if upto is not None:
        P = dict(P, layers=P["layers"][: upto + 1])
    nodes, inits = [], [numpy_helper.from_array(np.array(0, np.int8), "zp8"),
                        numpy_helper.from_array(np.array(0, np.int32), "zp32")]

    def scale(name, f):
        inits.append(numpy_helper.from_array(np.array(2.0 ** -f, np.float32), name))
        return name

    def qdq(x, f, tag):
        s = scale(f"{tag}_scale", f)
        nodes.append(helper.make_node("QuantizeLinear", [x, s, "zp8"], [f"{tag}_q"], name=f"{tag}_Q"))
        nodes.append(helper.make_node("DequantizeLinear", [f"{tag}_q", s, "zp8"], [f"{tag}_dq"], name=f"{tag}_DQ"))
        return f"{tag}_dq"

    x = qdq("mfcc", P["f_input"], "input")
    for i, L in enumerate(P["layers"]):
        tag = layer_tag(L["name"])
        if L["op"] == "avgpool":
            nodes.append(helper.make_node("GlobalAveragePool", [x], [f"{tag}_out"], name=tag))
            x = qdq(f"{tag}_out", L["f_out"], tag)
            continue
        inits.append(numpy_helper.from_array(L["w"].astype(np.int8), f"{tag}_w_q"))
        inits.append(numpy_helper.from_array(L["b"].astype(np.int32), f"{tag}_b_q"))
        nodes.append(helper.make_node("DequantizeLinear", [f"{tag}_w_q", scale(f"{tag}_w_scale", L["f_w"]), "zp8"],
                                      [f"{tag}_w"], name=f"{tag}_w_DQ"))
        nodes.append(helper.make_node("DequantizeLinear", [f"{tag}_b_q", scale(f"{tag}_b_scale", L["f_b"]), "zp32"],
                                      [f"{tag}_b"], name=f"{tag}_b_DQ"))
        kh, kw = L["w"].shape[2:]
        ph, pw = L["pad"]
        nodes.append(helper.make_node("Conv", [x, f"{tag}_w", f"{tag}_b"], [f"{tag}_conv"], name=f"{tag}_Conv",
                                      kernel_shape=[kh, kw], strides=list(L["stride"]), pads=[ph, pw, ph, pw],
                                      group=L["groups"], dilations=[1, 1]))
        y = f"{tag}_conv"
        if L["relu"]:
            nodes.append(helper.make_node("Relu", [y], [f"{tag}_relu"], name=f"{tag}_Relu"))
            y = f"{tag}_relu"
        x = qdq(y, L["f_out"], tag)
    nodes.append(helper.make_node("Identity", [x], ["logits"], name="logits"))
    g = helper.make_graph(nodes, "dscnn_kws_int8",
                          [helper.make_tensor_value_info("mfcc", TensorProto.FLOAT, [1, 1, C.N_FRAMES, C.N_MFCC])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, None)],
                          inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], producer_name="kws")
    m.ir_version = 8
    m = onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return m
