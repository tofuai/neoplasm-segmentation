from models import *
import torch
import numpy as np
import cv2
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = "/media/syan/163EAD8F3EAD6887/VinIF/Data/Models/210612_NeoSeg-DHA_352x352_1623509174.pth"
ONNX_FN = "blazeneo_clean"
OPSET = 9


def main(argv):
    model = NeoSeg(aggregation_type="DHA", auxiliary=False)
    # model = PraNet()
    # model = HarDNetMSEG()
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    # model.to(torch.device("cuda"))
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
        "{}.onnx".format(ONNX_FN),
        export_params=True,           # store the trained parameter weights inside the model file
        keep_initializers_as_inputs=True,
        do_constant_folding=True,     # whether to execute constant folding for optimization
        opset_version=OPSET,
        verbose=True,
        # dynamic_axes=dynamic_axes,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        input_names=input_names,      # the model's input names
        output_names=output_names     # the model's output names
    )


if __name__ == "__main__":
    main(sys.argv)
