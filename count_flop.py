import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import torch
from models import UNet, PraNet, HarDNetMSEG, NeoUNet, BlazeNeo

# from pthflops import count_ops

# # create segmentation model with pretrained encoder
# # model = BlazeNeo(aggregation="DHA", auxiliary=False)
# model = PraNet()
# model = model.to('cuda')

# count_ops(model, torch.rand(1, 3, 352, 352).to('cuda'))


import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    net = NeoUNet()
    macs, params = get_model_complexity_info(net, (3, 352, 352), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
