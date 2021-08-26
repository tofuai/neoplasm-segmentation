import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from models import UNet, PraNet, HarDNetMSEG, NeoUNet, BlazeNeo

from torchsummary import summary


# create segmentation model with pretrained encoder
# model = BlazeNeo(aggregation="DHA", auxiliary=False)
model = HarDNetMSEG()
model = model.to('cuda')

summary(model, (3, 352, 352))