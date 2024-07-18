import torch
import torch.nn as nn
import torch.nn.functional as F

import model.utils.correlation as correlation
import model.softsplat.softsplat as softsplat

class MotionEstimator(nn.Module):
    """Bi-directional optical flow estimator
    1) construct partial cost volume with the CNN features from the stage 2 of
    the feature pyramid;
    2) estimate bi-directional flows, by feeding cost volume, CNN features for
    both warped images, CNN feature and estimated flow from previous iteration.
    """

    def __init__(self, feature_dim):
        
        # 
        # Feature_dim: the channel of current pyramid feature. The last_feautre should be passed from pyramid layer to pyramid layer
        # so that it need to be doubled and interpolated to fint the next layer's shape and channel 
        super(MotionEstimator, self).__init__()
        # (4*2 + 1) ** 2 + 64 * 2 + 64 + 4 = 277
        self.conv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=277, out_channels=160,
                    kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=128,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=112,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels=112, out_channels=96,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer5 = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=64,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer6 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=4,
                    kernel_size=3, stride=1, padding=1))


    def forward(self, feat0, feat1, last_feat, last_flow):
        corr_fn=correlation.FunctionCorrelation
        feat0 = softsplat.FunctionSoftsplat(
                tenInput=feat0, tenFlow=last_flow[:, :2]*0.25*0.5,
                tenMetric=None, strType='average')
        feat1 = softsplat.FunctionSoftsplat(
                tenInput=feat1, tenFlow=last_flow[:, 2:]*0.25*0.5,
                tenMetric=None, strType='average')

        volume = F.leaky_relu(
                input=corr_fn(tenFirst=feat0, tenSecond=feat1),
                negative_slope=0.1, inplace=False)
        input_feat = torch.cat([volume, feat0, feat1, last_feat, last_flow], 1)
        feat = self.conv_layer1(input_feat)
        feat = self.conv_layer2(feat)
        feat = self.conv_layer3(feat)
        feat = self.conv_layer4(feat)
        feat = self.conv_layer5(feat)
        flow = self.conv_layer6(feat)
        
        return flow, feat