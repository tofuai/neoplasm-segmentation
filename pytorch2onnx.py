from models import DUNet
import torch
import numpy as np
import cv2
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = "/home/s/polyp/models/DUNet_190122.pth"
ONNX_PATH = "/home/s/polyp/models/DUNet_190122.onnx"
OPSET = 9


def main(argv):
    model = DUNet()
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    model.eval()

    x = torch.randn(1, 3, 352, 352, requires_grad=True)
    torch_out = model(x)

    input_names = ["input"]
    output_names = ["output"]

    # onnx generation
    torch.onnx.export(
        model,                        # model being run
        x,                      # model input (or a tuple for multiple inputs)
        # where to save the model (can be a file or file-like object)
        ONNX_PATH,
        export_params=True,           # store the trained parameter weights inside the model file
        keep_initializers_as_inputs=True,
        do_constant_folding=True,     # whether to execute constant folding for optimization
        opset_version=OPSET,
        verbose=True,
        # dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        input_names=input_names,      # the model's input names
        output_names=output_names     # the model's output names
    )


if __name__ == "__main__":
    main(sys.argv)
