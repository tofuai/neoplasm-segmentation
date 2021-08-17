import torch

from .module import Upsample, RFBblock, Head, LSC, IDA, DIA, DHA
from .hardnet_68 import hardnet


class BlazeNeo(torch.nn.Module):
    def __init__(self, aggregation="DHA", auxiliary=True):
        super().__init__()

        agg = {"LSC": LSC, "IDA": IDA, "DIA": DIA, "DHA": DHA}

        self.encoder = hardnet()

        # ---- Receptive Field Block like module ----
        self.rfb3 = RFBblock(320, 32)
        self.rfb4 = RFBblock(640, 32)
        self.rfb5 = RFBblock(1024, 32)

        # ---- Partial Decoder ----
        if auxiliary:
            self.agg1 = DHA(32)
            # self.agg1 = LSC(32)
            self.head_1 = Head(32, 1)
            self.upsample_1 = Upsample(scale_factor=8)
        
        self.agg2 = agg[aggregation](32)
        self.head_2 = Head(32, 3)
        self.upsample_2 = Upsample(scale_factor=8)
 
        self.auxiliary = auxiliary
        self.name = "blazeneo"

    def forward(self, x):
        _, _, _, x3, x4, x5 = self.encoder(x)

        x3_rfb = self.rfb3(x3)        # channel -> 32
        x4_rfb = self.rfb4(x4)        # channel -> 32
        x5_rfb = self.rfb5(x5)        # channel -> 32
        
        if self.auxiliary:
            agg1 = self.agg1(x5_rfb, x4_rfb, x3_rfb)
            h1 = self.head_1(agg1)
            S_1 = self.upsample_1(h1)

        agg2 = self.agg2(x5_rfb, x4_rfb, x3_rfb)
        h2 = self.head_2(agg2)
        S_2 = self.upsample_2(h2)

        if self.auxiliary:
            return S_1, S_2
        else:
            return S_2