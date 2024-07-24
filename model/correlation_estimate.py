import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ConvGRU import unitConvGRU as unitGRU
from model.softsplat import softsplat
from model.utils import correlation
from refine import conv




class coarse_motion_encoder_for_ini_optical_flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer0 = nn.Conv2d(6, 24,3, 1, 1)
        self.conv_layer1 = nn.Conv2d(24, 48,3, 1, 1)
        self.conv_layer2 = nn.Conv2d(48, 64,3, 1, 1)
        self.conv_layer3 = nn.Conv2d(64, 32,3, 1, 1)
        self.optical_flow_mask = nn.Conv2d(32, 6, 3, 1, 1)
    def forward(self, x):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        # gt = x[:, 6:]

        feat = self.conv_layer0(torch.cat([img0, img1], dim=1))
        feat = self.conv_layer1(feat)
        feat = self.conv_layer2(feat)
        feat = self.conv_layer3(feat)
        flow_mask = self.optical_flow_mask(feat)

        flow_forward = flow_mask[:,:2]
        flow_backward = flow_mask[:,2:4]
        mask_f = flow_mask[:, 4]
        mask_b = flow_mask[:, 5]

        return flow_forward, flow_backward, mask_f, mask_b


class IF_Recurrent_Image_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, pyramid="image"):
        super().__init__()
        self.pyramid = pyramid
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru_unit = unitGRU(self.in_channels, self.out_channels)
        #
        self.conv_f_feature = nn.Sequential(
            nn.Conv2d(in_channels= 3 + self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            conv(self.out_channels, self.out_channels),
            conv(self.out_channels, self.out_channels),
            conv(self.out_channels, self.out_channels)
        )
        self.conv_b_feature = nn.Sequential(
            nn.Conv2d(in_channels= 3 + self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            conv(self.out_channels, self.out_channels),
            conv(self.out_channels, self.out_channels),
            conv(self.out_channels, self.out_channels)
        )
        self.conv_f_delta_f_mask = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=self.out_channels // 2, out_channels=6, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.conv_b_delta_f_mask = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=self.out_channels // 2, out_channels=6, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.transformed_image_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    def forward_delta_flow_block(self, forward_warped_image_this_layer, forward_temporal_feature_this_layer, ds_image_1):
        forward_warped_feature_this_layer = self.conv_f_feature(torch.cat([forward_warped_image_this_layer,forward_temporal_feature_this_layer], dim=1))
        delta_f_with_mask = self.conv_f_delta_mask(torch.cat([forward_warped_feature_this_layer, ds_image_1], dim=1))
        return delta_f_with_mask, forward_warped_feature_this_layer

    def backward_delta_flow_block(self, back_warped_image_this_layer, back_temporal_feature_this_layer, ds_image_0):
        """
        back_warped_image_this_layer: B*C* H/8 * W/8
        back_temporal_feature_this_layer: B*C* H/8 * W/8
        ds_image_0 B * 3 * H/8 * W/8
        ==>
        delta_b: B * [:2] * H/4 * W/4
        Mask: B* 1 * H/4 * W/4
        """
        back_warped_feature_this_layer = self.conv_b_feature(torch.cat([back_warped_image_this_layer,back_temporal_feature_this_layer], dim=1)
        delta_b_with_mask = self.conv_b_delta_f_mask(torch.cat([back_warped_feature_this_layer, ds_image_0], dim=1)
        return delta_b_with_mask, back_warped_feature_this_layer

    def forward(self, ds_image_0, ds_image_1,
                integrated_forward_flow, integrated_backward_flow,
                forward_temporal_feature_this_layer,
                backward_temporal_feature_this_layer):
        # The integrated flow should be [B * 4 * H * W] where first two dimension is for forward optical flow and

        # current plan: only use cnn to estimate the delta optical flow

        # adjust the resolution of optical flow
        flow_forward = F.interpolate(integrated_forward_flow, scale_factor=2, mode="bilinear",
                                     align_corners=False) * 2
        flow_backward = F.interpolate(integrated_backward_flow, scale_factor=2, mode="bilinear",
                                      align_corners=False) * 2

        if self.pyramid == "image":
            forward_warped_image = warplayer.warp(ds_image_0, flow_forward)
            backward_warped_image = warplayer.warp(ds_image_1, flow_backward)

        elif self.pyramid == "feature":
            raise NotImplementedError("Feature pyramid is not implemented yet.")

        else:
            raise ValueError("Invalid value for self.pyramid: {}".format(self.pyramid))

        delta_f_with_mask, forward_ds_features = self.forward_delta_flow_block(
            forward_warped_image, forward_temporal_feature_this_layer, ds_image_1)
        delta_b_with_mask, backward_ds_features = self.backward_delta_flow_block(
            backward_warped_image, backward_temporal_feature_this_layer, ds_image_0)


        mask_f = delta_f_with_mask[:, 2:]
        mask_b = delta_b_with_mask[:, 2:]
        flow_forward = flow_forward + delta_f_with_mask[:, 2]
        flow_backward = flow_backward + delta_b_with_mask[:, 2]



        merged = mask_b * backward_warped_image + mask_f * forward_warped_image

        return merged, flow_forward, flow_backward, mask_f, mask_b, forward_ds_features, backward_ds_features
        # the forward_ds_features and backward_ds_features are for the unet.




class prediction_for_intermediate_frame(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        # prediction
        refine = self.pred(x)
        refine_res = torch.sigmoid(refine[:, :3]) * 2 - 1
        refine_mask0 = torch.sigmoid(refine[:, 3:4])
        refine_mask1 = torch.sigmoid(refine[:, 4:5])
        merged_img = (warped_img0 * refine_mask0 * (1 - time_period) + \
                      warped_img1 * refine_mask1 * time_period)
        merged_img = merged_img / (refine_mask0 * (1 - time_period) + \
                                   refine_mask1 * time_period)
        interp_img = merged_img + refine_res
        interp_img = torch.clamp(interp_img, 0, 1)


class delta_Optical_Flow_Estimator(nn.Module):
    def __init__(self, pyramid_feature_channel=32 ):
        super().__init__()
        self.corr_fn = correlation.FunctionCorrelation
        self.flow_compute = nn.Sequential(
            nn.Conv2d(in_channels=pyramid_feature_channel*3, out_channels=pyramid_feature_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=pyramid_feature_channel*2, out_channels=pyramid_feature_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=pyramid_feature_channel*2, out_channels=pyramid_feature_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=pyramid_feature_channel*2, out_channels=pyramid_feature_channel, kernel_size=3, stride=1, padding=1, bias=False),

            nn.Conv2d(in_channels=pyramid_feature_channel * 2, out_channels=pyramid_feature_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=pyramid_feature_channel, out_channels=pyramid_feature_channel * 0.5, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=pyramid_feature_channel*0.5, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1)
        )
    # def flow_compute(self, warped_feature, temporal_feature, volume):


    def correlation_pack_of_two_features(self, bi_flow_previous, feature_0, feature_1):
        # The bi_flow and features should at the same
        # So the bi_flow_previous is the direct output of the previous layer

        bi_flow = F.interpolate(
                    input=bi_flow_previous, scale_factor=2,
                    mode="bilinear", align_corners=False) * 2
        warped_feat0 = softsplat.FunctionSoftsplat(
                tenInput=feature_0, tenFlow=bi_flow[:, :2],
                tenMetric=None, strType='average')
        warped_feat1 = softsplat.FunctionSoftsplat(
                tenInput=feature_1, tenFlow=bi_flow[:, 2:],
                tenMetric=None, strType='average')
        volume = F.leaky_relu(
                input= self.corr_fn(tenFirst=warped_feat0, tenSecond=warped_feat1),
                negative_slope=0.1, inplace=False)
        return warped_feat0, warped_feat1, volume


    def forward(self, feature_0, feature_1, bi_flow_previous, forward_temporal_feature, backward_temporal_feature):
        # Do not pass cost volume here

        warped_feature0, warped_feature1, volume = self.correlation_pack_of_two_features(bi_flow_previous, feature_0, feature_1)

        # Forward compute
        delta_bi_flow_forward = self.flow_compute(warped_feature0, forward_temporal_feature, volume)

        # Backward compute 
        delta_bi_flow_backward = self.flow_compute(warped_feature1, backward_temporal_feature, volume)

        delta_bi_flow = torch.cat([delta_bi_flow_forward, delta_bi_flow_backward], dim=1)
        
        return delta_bi_flow
        
        
class predict(nn.Module):
    def __init__(self):
        self.pas = None
        self.delta_OF_Estimator = nn.ModuleList([
            delta_Optical_Flow_Estimator(pyramid_feature_channel=32),
            delta_Optical_Flow_Estimator(pyramid_feature_channel=64),
            delta_Optical_Flow_Estimator(pyramid_feature_channel=128)]
        )
    def splat(self, feature_0, feature_1, bi_flow):
        warped_feat0 = softsplat.FunctionSoftsplat(
            tenInput=feature_0, tenFlow=bi_flow[:, :2],
            tenMetric=None, strType='average')
        warped_feat1 = softsplat.FunctionSoftsplat(
            tenInput=feature_1, tenFlow=bi_flow[:, 2:],
            tenMetric=None, strType='average')
        return warped_feat0, warped_feat1

    def forward(self, image0, image1,  image0_feature_pyramid, image1_feature_pyramid, forward_feature_pyramid, backward_feature_pyramid):
        N, C, H, W = image0.size()
        bi_flow = torch.zeros(
                        (N, 4, H // 4, W //4 )
                        ).to(image0.device)
        init_feat = torch.zeros(
                        (N, 64, H // 4, W // 4)
                        ).to(image0.device)

        for i in range(3):
            forward_temporal_feature = forward_feature_pyramid[i]
            backward_temporal_feature = backward_feature_pyramid[i]

            feature_0 = image0_feature_pyramid[i]
            feature_1 = image1_feature_pyramid[i]

            delta_bi_flow = self.delta_Optical_Flow_Estimator[i](feature_0, feature_1, bi_flow, forward_temporal_feature, backward_temporal_feature)
            bi_flow = bi_flow + delta_bi_flow

        warped_feature0, warped_feature1 = self.splat(feature_0, feature_1, bi_flow)
        mask_0 = self.mask_estimator(torch.cat((warped_feature0, feature_0), dim=1))

        mask_1 = self.mask_estimator(torch.cat((warped_feature1, feature_1), dim=1))

        warped_image0, warped_image1 = self.splat(image0, image1, bi_flow)

        final_output = mask_0*warped_image0 + mask_1*warped_image1

        return final_output




        
        