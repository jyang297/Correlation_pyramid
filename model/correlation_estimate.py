import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ConvGRU import unitConvGRU as unitConvGRU
from model.ConvGRU import PyramidFBwardExtractor
from model.softsplat import softsplat
from model.utils import correlation
# from refine import conv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvGRUFeatures(nn.Module):
    """
    Student ConvGRU feature extractor
    """

    def __init__(self, hidden_dim=64, pyramid="image"):
        super().__init__()
        # current encoder: all frames ==> all features
        self.pyramid = pyramid
        self.hidden_dim = hidden_dim
        self.output_plane = self.hidden_dim //2
        # self.img2Fencoder = PyramidFBwardExtractor(in_plane=3, hidden_pyramid=self.hidden_dim,
        #                                               pyramid=self.pyramid)

        self.hidden_dim_d0 = hidden_dim
        self.hidden_dim_d2 = hidden_dim * 2
        self.hidden_dim_d4 = hidden_dim * 4
        self.forwardgru_d0 = unitConvGRU(hidden_dim=self.hidden_dim_d0, input_dim=self.hidden_dim_d0)
        self.backwardgru_d0 = unitConvGRU(hidden_dim=self.hidden_dim_d0, input_dim=self.hidden_dim_d0)
        self.forwardgru_d2 = unitConvGRU(hidden_dim=self.hidden_dim_d2, input_dim=self.hidden_dim_d2)
        self.backwardgru_d2 = unitConvGRU(hidden_dim=self.hidden_dim_d2, input_dim=self.hidden_dim_d2)
        self.forwardgru_d4 = unitConvGRU(hidden_dim=self.hidden_dim_d4, input_dim=self.hidden_dim_d4)
        self.backwardgru_d4 = unitConvGRU(hidden_dim=self.hidden_dim_d4, input_dim=self.hidden_dim_d4)



    def forward(self, allframes, fallfeatures_d0, fallfeatures_d2, fallfeatures_d4):
        # aframes = allframes_N.view(b,n*c,h,w)
        # Output: BNCHW
        fcontextlist_d0 = []  # c = 64
        bcontextlist_d0 = []  # c = 64
        fcontextlist_d2 = []  # 2c = 128
        bcontextlist_d2 = []  # 2c = 128
        fcontextlist_d4 = []  # 4c = 256
        bcontextlist_d4 = []  # 4c = 256
        # fallfeatures_d0, fallfeatures_d2, fallfeatures_d4 = self.img2Fencoder(allframes, pyramid="image")

        b, _, h, w = allframes.size()

        # forward GRU
        # Method A: zero initialize Hiddenlayer
        if self.pyramid == "image":
            forward_hidden_initial_d0 = torch.zeros((b, self.hidden_dim_d0, h, w), device=device)
            backward_hidden_initial_d0 = torch.zeros((b, self.hidden_dim_d0, h, w), device=device)
        forward_hidden_initial_d2 = torch.zeros((b, self.hidden_dim_d2, h // 2, w // 2), device=device)
        backward_hidden_initial_d2 = torch.zeros((b, self.hidden_dim_d2, h // 2, w // 2), device=device)
        forward_hidden_initial_d4 = torch.zeros((b, self.hidden_dim_d4, h // 4, w // 4), device=device)
        backward_hidden_initial_d4 = torch.zeros((b, self.hidden_dim_d4, h // 4, w // 4), device=device)
        # n=4
        # I skipped the 0 -> first image

        if self.pyramid == "image":
            for i in range(0, 4):
                if i == 0:
                    # for d0 layer
                    fhidden = self.forwardgru_d0(forward_hidden_initial_d0, fallfeatures_d0[i])
                    bhidden = self.backwardgru_d0(backward_hidden_initial_d0, fallfeatures_d0[-i - 1])

                else:
                    fhidden = self.forwardgru_d0(fhidden, fallfeatures_d0[i])
                    bhidden = self.backwardgru_d0(bhidden, fallfeatures_d0[-i - 1])
                    fcontextlist_d0.append(fhidden)
                    bcontextlist_d0.append(bhidden)

        for i in range(0, 4):
            if i == 0:
                # for d2 layer
                fhidden = self.forwardgru_d2(forward_hidden_initial_d2, fallfeatures_d2[i])
                bhidden = self.backwardgru_d2(backward_hidden_initial_d2, fallfeatures_d2[-i - 1])
            else:
                fhidden = self.forwardgru_d2(fhidden, fallfeatures_d2[i])
                bhidden = self.backwardgru_d2(bhidden, fallfeatures_d2[-i - 1])
                fcontextlist_d2.append(fhidden)
                bcontextlist_d2.append(bhidden)
        for i in range(0, 4):
            if i == 0:
                # for d4 layer
                fhidden = self.forwardgru_d4(forward_hidden_initial_d4, fallfeatures_d4[i])
                bhidden = self.backwardgru_d4(backward_hidden_initial_d4, fallfeatures_d4[-i - 1])
            else:
                fhidden = self.forwardgru_d4(fhidden, fallfeatures_d4[i])
                bhidden = self.backwardgru_d4(bhidden, fallfeatures_d4[-i - 1])
                fcontextlist_d4.append(fhidden)
                bcontextlist_d4.append(bhidden)
        
        return fcontextlist_d0, fcontextlist_d2, fcontextlist_d4, bcontextlist_d0, bcontextlist_d2, bcontextlist_d4

        # return forwardFeature, backwardFeature
        # Now iterate through septuplet and get three inter frames


class delta_Optical_Flow_Estimator(nn.Module):
    def __init__(self, pyramid_feature_channel=32 ):
        super().__init__()
        self.corr_fn = correlation.FunctionCorrelation
        self.flow_compute = nn.Sequential(
            nn.Conv2d(in_channels=pyramid_feature_channel*2+81, out_channels=pyramid_feature_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=pyramid_feature_channel*2, out_channels=pyramid_feature_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=pyramid_feature_channel*2, out_channels=pyramid_feature_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=pyramid_feature_channel*2, out_channels=pyramid_feature_channel, kernel_size=3, stride=1, padding=1, bias=False),

            nn.Conv2d(in_channels=pyramid_feature_channel, out_channels=pyramid_feature_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=pyramid_feature_channel, out_channels=pyramid_feature_channel //2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=pyramid_feature_channel //2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1)
        )
    # def flow_compute(self, warped_feature, temporal_feature, volume):


    def correlation_pack_of_two_features(self, bi_flow_previous, feature_0, feature_1):
        # The bi_flow and features should at the same
        # So the bi_flow_previous is the direct output of the previous layer

        bi_flow = F.interpolate(
                    input=bi_flow_previous, scale_factor=2,
                    mode="bilinear", align_corners=False) * 2
        bi_flow = bi_flow.contiguous()
        
        warped_feat0 = softsplat.FunctionSoftsplat(
                tenInput=feature_0, tenFlow=bi_flow[:, :2].contiguous(),
                tenMetric=None, strType='average')
        warped_feat1 = softsplat.FunctionSoftsplat(
                tenInput=feature_1, tenFlow=bi_flow[:, 2:].contiguous(),
                tenMetric=None, strType='average')
        volume = F.leaky_relu(
                input= self.corr_fn(tenFirst=warped_feat0, tenSecond=warped_feat1),
                negative_slope=0.1, inplace=False)
        return warped_feat0, warped_feat1, volume, bi_flow


    def forward(self, feature_0, feature_1, bi_flow_previous, forward_temporal_feature, backward_temporal_feature):
        # Do not pass cost volume here


        warped_feature0, warped_feature1, volume, bi_flow_inp = self.correlation_pack_of_two_features(bi_flow_previous, feature_0, feature_1)

        # Forward compute
        tensor_delta_flow_f = torch.cat([warped_feature0, forward_temporal_feature, volume], dim=1)
        delta_bi_flow_forward = self.flow_compute(tensor_delta_flow_f)
        # Backward compute 
        delta_bi_flow_backward = self.flow_compute(torch.cat([warped_feature1, backward_temporal_feature, volume], dim=1))

        delta_bi_flow = torch.cat([delta_bi_flow_forward, delta_bi_flow_backward], dim=1)
        
        bi_flow = bi_flow_inp + delta_bi_flow
        
        return bi_flow
        
        
class predict(nn.Module):
    def __init__(self, pyramid_channel=32):
        super().__init__()
        
        self.pas = None
        self.pyramid_channel=pyramid_channel
        self.delta_OF_Estimator = nn.ModuleList([
            delta_Optical_Flow_Estimator(pyramid_feature_channel=self.pyramid_channel*4),
            delta_Optical_Flow_Estimator(pyramid_feature_channel=self.pyramid_channel*2),
            delta_Optical_Flow_Estimator(pyramid_feature_channel=self.pyramid_channel)]
        )
        self.mask_estimator = nn.Sequential(
            nn.Conv2d(self.pyramid_channel*2, self.pyramid_channel, 3, 1,1),
            nn.Conv2d(self.pyramid_channel, self.pyramid_channel//2, 3, 1,1),
            nn.Conv2d(self.pyramid_channel//2, self.pyramid_channel//2, 3, 1,1),
            nn.Conv2d(self.pyramid_channel//2, 4, 3,1,1),
            nn.Conv2d(4,1,3,1,1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    def splat(self, feature_0, feature_1, bi_flow):
        warped_feat0 = softsplat.FunctionSoftsplat(
            tenInput=feature_0, tenFlow=bi_flow[:, :2].contiguous(),
            tenMetric=None, strType='average')
        warped_feat1 = softsplat.FunctionSoftsplat(
            tenInput=feature_1, tenFlow=bi_flow[:, 2:].contiguous(),
            tenMetric=None, strType='average')
        return warped_feat0, warped_feat1

    def forward(self, image0, image1,  image0_feature_pyramid, image1_feature_pyramid, forward_feature_pyramid, backward_feature_pyramid):
        N, C, H, W = image0.size()
        bi_flow = torch.zeros(
                        (N, 4, H // (4*2), W //(4*2) )
                        ).to(image0.device)
        init_feat = torch.zeros(
                        (N, 64, H // 4, W // 4)
                        ).to(image0.device)

        for i in range(3):
            forward_temporal_feature = forward_feature_pyramid[i]
            backward_temporal_feature = backward_feature_pyramid[i]

            feature_0 = image0_feature_pyramid[i]
            feature_1 = image1_feature_pyramid[i]

            bi_flow = self.delta_OF_Estimator[i](feature_0, feature_1, bi_flow, forward_temporal_feature, backward_temporal_feature)
         

        warped_feature0, warped_feature1 = self.splat(feature_0, feature_1, bi_flow)
        mask_0 = self.mask_estimator(torch.cat((warped_feature0, feature_0), dim=1))

        mask_1 = self.mask_estimator(torch.cat((warped_feature1, feature_1), dim=1))

        warped_image0, warped_image1 = self.splat(image0, image1, bi_flow)

            
        merged_img = warped_image0 * mask_0 * + warped_image1 * mask_1 
        merged_img = merged_img / (mask_0  + mask_1 )
        
        return merged_img




        
        