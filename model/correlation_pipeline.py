import torch
import torch.nn as nn

import model.laplacian as modelLap
from model.warplayer import warp
from model.refine import Unet_for_3Pyramid as unet3P
from model.loss import *
from model.myLossset import *
from model.ConvGRU import PyramidFBwardExtractor as Pyramid_direction_extractor

import model.correlation_estimate as cest
from model.myLossset import CensusLoss as census
import model.STloss as ST


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
c = 48


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )







class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramid = "image"
        self.hidden = 32
        self.gru = cest.ConvGRUFeatures(self.hidden)
        self.img2Fencoder = Pyramid_direction_extractor(in_plane=3, hidden_pyramid=self.hidden,
                                                        pyramid=self.pyramid)
        # self.img2Bencoder = Pyramid_direction_extractor(in_plane=3, hidden_pyramid=self.hidden,
        # pyramid=self.pyramid)
        self.predict = cest.predict(self.hidden)
        # self.interpolate = unet3P(hidden_dim=self.hidden, shift_dim=self.hidden)
        self.decoder = nn.Sequential()
        self.epsilon = 1e-6
        self.loss_census = census()

    def forward(self, allframes, training_flag=True):
        # allframes 0<1>2<3>4<5>6

        Sum_loss_context = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_tea_pred = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_mse = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')

        output_allframes = []
        output_teacher = []
        flow_list = []
        mask_list = []
        fallfeatures_d0, fallfeatures_d2, fallfeatures_d4 = self.img2Fencoder(allframes, pyramid="image")

        fcontextlist_d0, fcontextlist_d2, fcontextlist_d4, bcontextlist_d0, bcontextlist_d2, bcontextlist_d4 = self.gru(allframes, fallfeatures_d0, fallfeatures_d2, fallfeatures_d4)
        forward_feature_pyramid = []
        backward_feature_pyramid = []
        image_feature_pyramid = []
        for i in range(4):
            tmp_list = [fallfeatures_d4[i], fallfeatures_d2[i], fallfeatures_d0[i]]
            image_feature_pyramid.append(tmp_list)
        for i in range(3):
            tmp_list = [fcontextlist_d4[i], fcontextlist_d2[i], fcontextlist_d0[i]]
            forward_feature_pyramid.append(tmp_list)
            tmp_list = [bcontextlist_d4[-i-1], bcontextlist_d2[-i-1], bcontextlist_d0[-i-1]]
            backward_feature_pyramid.append(tmp_list)
            
            
                
       

        for i in range(3):
            if training_flag:
                image0 = allframes[:, 6 * i:6 * i + 3]
                gt = allframes[:, 6 * i + 3:6 * i + 6]
                image1 = allframes[:, 6 * i + 6:6 * i + 9]
                
        
            predictimage = self.predict(image0, image1,  image_feature_pyramid[i], image_feature_pyramid[i+1], forward_feature_pyramid[i], backward_feature_pyramid[i])

            # The [i][-1] is supposed to pick up the d0 level
 


 

            # Use Unet to create the three interpolated frame

            # featureUnet = self.interpolate(ori_img0_feature, ori_img1_feature, warped_fimg0_d0,
                                        #    warped_fimg1_d0, f_att_d0,
                                        #    f_att_d2, f_att_d4, b_att_d0, b_att_d2, b_att_d4)
            # flow, mask, merged, flow_teacher, merged_teacher, loss_tea_pred = self.interpolate(allframes)
            # predictimage = self.decoder(featureUnet)

            # Start loss computation
            loss_pred = torch.mean(
                torch.sqrt(torch.pow((predictimage - gt), 2) + self.epsilon ** 2)) + self.loss_census(
                predictimage, gt)
            loss_mse = ((predictimage - gt) ** 2).detach()
            loss_mse = loss_mse.mean()

            # loss_tea = 0
            merged_teacher = predictimage * 0  # not used. just to avoid error
            flow_teacher = predictimage * 0  # not used. just to avoid error
            flow = predictimage * 0
            mask = predictimage * 0
            mask_list = [flow_teacher, flow_teacher, flow_teacher]
            flow_list = [flow_teacher, flow_teacher, flow_teacher]

            # =======================================================
            Sum_loss_context += loss_pred
            Sum_loss_mse += loss_mse
            Sum_loss_tea_pred += 0
            output_allframes.append(image0)
            output_teacher.append(image0)
            # output_allframes.append(merged[2])
            output_allframes.append(predictimage)
            flow_list.append(flow)
            mask_list.append(mask)

            # The way RIFE compute prediction loss and 
            # loss_l1 = (self.lap(merged[2], gt)).mean()
            # loss_tea = (self.lap(merged_teacher, gt)).mean()

        img6 = allframes[:, -3:]
        output_allframes.append(img6)
        output_teacher.append(img6)
        output_allframes_tensors = torch.stack(output_allframes, dim=1)
        output_teacher_tensors = torch.stack(output_teacher, dim=1)

        # Dummy output
        flow_teacher_list = flow_list
        loss_dist = 0

        return flow_list, mask_list, output_allframes_tensors, flow_teacher_list, output_teacher_tensors, Sum_loss_tea_pred, Sum_loss_context, Sum_loss_mse, loss_dist
