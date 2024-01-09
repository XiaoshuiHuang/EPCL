import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from .transformer import EPCLEncoder, EPCLPreEncoder
from .util_modules import PointNetFeaturePropagation

class MinkResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(MinkResBlock, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False,
            dimension=3,
        )
        self.norm1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=3,
        )

        self.norm2 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out


class EPCLDetection(nn.Module):

    def __init__(
        self,
        pre_encoder,
        encoder,
        encoder_dim=256,
        voxel_size=None
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.embdding_layer = nn.Sequential(
            ME.MinkowskiConvolution(
                3, 64, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64, momentum=0.1),
            ME.MinkowskiReLU(inplace=True),
        )
        self.downsampling_layer_2 = nn.Sequential(
            ME.MinkowskiConvolution(
                64, 128, kernel_size=3, stride=2, dimension=3), 
            ME.MinkowskiBatchNorm(128, momentum=0.1),
            ME.MinkowskiReLU(inplace=True),
            MinkResBlock(in_channels=128, out_channels=128)
        )
        self.downsampling_layer_4 = nn.Sequential(
            ME.MinkowskiConvolution(
                128, 256, kernel_size=3, stride=2, dimension=3),  
            ME.MinkowskiBatchNorm(256, momentum=0.1),
            ME.MinkowskiReLU(inplace=True),
            MinkResBlock(in_channels=256, out_channels=256)
        )
        self.downsampling_layer_8 = nn.Sequential(
            ME.MinkowskiConvolution(
                256, 512, kernel_size=3, stride=2, dimension=3),  
            ME.MinkowskiBatchNorm(512, momentum=0.1),
            ME.MinkowskiReLU(inplace=True),
            MinkResBlock(in_channels=512, out_channels=512)
        )
        self.upsampling_layer_4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(512, 256, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(256, momentum=0.1),
            ME.MinkowskiReLU(inplace=True),
            MinkResBlock(in_channels=256, out_channels=256)
        )
        self.upsampling_layer_2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(256, 128, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(128, momentum=0.1),
            ME.MinkowskiReLU(inplace=True),
            MinkResBlock(in_channels=128, out_channels=128)
        )
        self.out_layer = nn.Sequential(
            ME.MinkowskiConvolution(128, 64, kernel_size=1, bias=False, dimension=3),
            ME.MinkowskiBatchNorm(64, momentum=0.1),
            ME.MinkowskiReLU(inplace=True),
        )
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.num_point_features = 64
        self.encoder_dim = encoder_dim

    def run_encoder(self, xyz, features):
        # Fixed number of tokens
        pre_enc_xyz, pre_enc_features, pre_enc_inds, ori_xyz = self.pre_encoder(
            xyz, features)

        pre_enc_features = pre_enc_features.permute(2, 0, 1)
        # use frozen clip transformer
        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz*self.voxel_size
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(
                pre_enc_inds, 1, enc_inds.type(torch.int64))
        return pre_enc_xyz, enc_features, enc_inds, ori_xyz

    def forward(self, input_dict, encoder_only=False):

        out_dict = dict()
        x = input_dict['sp_tensor']
        # downsampling
        feature_map = []
        x = self.embdding_layer(x)
        x = self.downsampling_layer_2(x)
        feature_map.append(x)
        x = self.downsampling_layer_4(x)
        feature_map.append(x)
        x = self.downsampling_layer_8(x)
            
        input_xyz = x.C
        input_features = x.F
        enc_xyz, enc_features, enc_inds, ori_xyz = self.run_encoder(input_xyz, input_features)
        # TODO:put init
        # Restore sparse convolutional features
        upsamping = PointNetFeaturePropagation(
            self.encoder_dim, [512])  # out_put dim
        ori_size_feature = []
        for i, one_batch_ori_xyz in enumerate(ori_xyz):
            one_batch_ori_size_features = upsamping(
                one_batch_ori_xyz, enc_xyz[i:i+1, ...], None, enc_features.permute(1, 0, 2)[i:i+1, ...])
            ori_size_feature.append(
                one_batch_ori_size_features.permute(0, 2, 1).squeeze(0))  # [N,C]

        features = torch.cat(ori_size_feature, dim=0)
        new_features = features + x.F
        
        output_sp = ME.SparseTensor(features=new_features,
                                    coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        # upsampling
        output_sp = self.upsampling_layer_4(output_sp)
        output_sp = output_sp + feature_map[-1]
        output_sp = self.upsampling_layer_2(output_sp)
        output_sp = output_sp + feature_map[-2]
        output_sp = self.out_layer(output_sp)
        out_dict['sp_tensor'] = output_sp
        
        return out_dict

def build_preencoder(args):

    preencoder = EPCLPreEncoder(
        num_group=args.PREENC_NPOINTS,
        group_size=args.GROUPS_SIZE,  
        enc_dim=args.ENC_DIM,
        in_channels=args.PREENCODER_DIM
    )
    
    return preencoder


def build_encoder(args):
   
    encoder = EPCLEncoder(embed_dim=args.ENC_DIM)

    return encoder

def build_epcl(args):
    
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    model = EPCLDetection(
                pre_encoder,
                encoder,
                encoder_dim=args.ENC_DIM,
                voxel_size=args.VOXEL_SIZE
            )
      
    return model
