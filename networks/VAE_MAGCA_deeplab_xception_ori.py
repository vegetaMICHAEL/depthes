# -*- encoding: utf-8 -*-
# author:lmolhw
# datetime:2021-9-27 15:09

"""
文件说明：
        给MAGCA添加VAE分支
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import sys

sys.path.append('/data1/lhw/workspace/UniFuse-Unidirectional-Fusion/UniFuse')
from networks.modality.bases import dct
from networks.horizon_refinement.attention import TransEn
from networks.barchnorm import SynchronizedBatchNorm2d
from networks.context_block import MultiAspectGCAttention

sys.path.append('/data1/lhw/workspace/UniFuse-Unidirectional-Fusion/UniFuse/s-vae-pytorch')
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

BatchNorm2d = SynchronizedBatchNorm2d


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self)._init_()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True,
                 is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
            rep.append(BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))

        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, inplanes=3, os=16, pretrained=False):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            is_last=True)

        # Middle flow
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                            grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                            grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                            grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                            grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                            grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                            grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True,
                             grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn3 = BatchNorm2d(1536)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn4 = BatchNorm2d(1536)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn5 = BatchNorm2d(2048)

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in model_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=1, os=16, pretrained=False, freeze_bn=False, _print=True, ehc=False,
                 vae=True, distribution='normal', freeze_ASPP=False):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Backbone: Xception")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.xception_features = Xception(nInputChannels, os, pretrained)
        self.context_modules = MultiAspectGCAttention(inplanes=256, ratio=0.0625, headers=4, att_scale=True)
        # ASPP
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(2048, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(2048, 256, dilation=dilations[2])
        self.aspp4 = ASPP_module(2048, 256, dilation=dilations[3])

        self.relu = nn.ReLU()
        self.vae_relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        '''EHC Param'''
        # self.horizon_refine = TransEn(1024, 256)
        # self.basis = nn.Parameter(dct(64, 1))

        self.ehc = ehc
        self.vae = vae
        if self.vae:
            self.z_dim, self.activation, self.distribution = 256, F.relu, distribution
            self.h_dim = 128
            # 2 hidden layers encoder
            self.fc_e0 = nn.Linear(256, self.h_dim * 2)
            self.fc_e1 = nn.Linear(self.h_dim * 2, self.h_dim)

            if self.distribution == 'normal':
                # compute mean and std of the normal distribution
                self.fc_mean = nn.Linear(self.h_dim, self.z_dim)
                self.fc_var =  nn.Linear(self.h_dim, self.z_dim)
            elif self.distribution == 'vmf':
                # compute mean and concentration of the von Mises-Fisher
                self.fc_mean = nn.Linear(self.h_dim, self.z_dim)
                self.fc_var = nn.Linear(self.h_dim, 1)
            else:
                raise NotImplemented
                # 2 hidden layers decoder
            self.fc_d0 = nn.Linear(self.z_dim, self.h_dim)
            self.fc_d1 = nn.Linear(self.h_dim, self.h_dim * 2)
            self.fc_logits = nn.Linear(self.h_dim * 2, 256)

        if freeze_bn:
            self._freeze_bn()
        if freeze_ASPP:
            self.freeze_ASPP_encoder()


    def ASPP_encoder(self, input):
        x, low_level_features = self.xception_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)

        return x, low_level_features

    def DeepLab_decoder(self, low_level_features, x, input):

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_ASPP_encoder(self):
        for param in self.xception_features.parameters():
            param.requires_grad = False
        for param in self.aspp1.parameters():
            param.requires_grad = False
        for param in self.aspp2.parameters():
            param.requires_grad = False
        for param in self.aspp3.parameters():
            param.requires_grad = False
        for param in self.aspp4.parameters():
            param.requires_grad = False
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.context_modules.parameters():
            param.requires_grad = False


    def forward(self, input):
        # Base ASPP Encoder
        x, low_level_features = self.ASPP_encoder(input['rgb'].cuda())  # [batch, 256, 128, 256]
        # x, low_level_features = self.ASPP_encoder(input)  # [batch, 256, 128, 256]
        context, tmp = self.context_modules(x)
        context += x
        x = self.relu(context)
        q_z = None
        p_z = None
        if self.vae:
            y, _ = self.ASPP_encoder(input['gt_depth'].repeat([1, 3, 1, 1]).type(torch.cuda.FloatTensor))
            _, tmp_y = self.context_modules(y)
            tmp, tmp_y = tmp.reshape([-1, 256]), tmp_y.reshape([-1, 256])
            feat_1 = tmp + tmp_y  # [batch, 256]
            feat_2 = tmp
            z_mean1, z_var1 = self.vae_encode(feat_1, feat_1)
            z_mean2, z_var2 = self.vae_encode(feat_2, feat_2)  # z_maen from x, z_var from gt
            q_z, _ = self.reparameterize(z_mean1, z_var1)
            _, p_z = self.reparameterize(z_mean2, z_var2)
            z = q_z.rsample()
            # z = self.vae_decode(z).reshape([-1, 256, 1, 1])
            # vae_tmp = x * z * 0.2
            #vae_tmp += x
            N, C2, H, W = x.shape
            vae_tmp = self.vae_decode(z).expand(-1, -1, H, W)

            
            #x = self.relu(vae_tmp)
            vae_tmp = self.relu(vae_tmp)
            x = self.relu(x)
            # x = self.relu(vae_tmp)
        feat = self.DeepLab_decoder(low_level_features, vae_tmp, input['rgb'].cuda())
        feat = feat*0.1
        feat += self.DeepLab_decoder(low_level_features, x, input['rgb'].cuda())
        #x = self.DeepLab_decoder(low_level_features, x, input['rgb'].cuda())
        # x = self.DeepLab_decoder(low_level_features, x, input)

        return {'pred': x, 'q_z': q_z, 'p_z':p_z}

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def infer(self, input):
        x, low_level_features = self.ASPP_encoder(input['rgb'].cuda())  # [batch, 256, 128, 256]
        context, tmp = self.context_modules(x)
        context += x
        x = self.relu(context)
        if self.vae:
            tmp = tmp.reshape([-1, 256])
            feat_2 = tmp
            z_mean2, z_var2 = self.vae_encode(feat_2, feat_2)  # z_maen from x, z_var from gt
            q_z, p_z = self.reparameterize(z_mean2, z_var2)
            z = q_z.rsample()
            #z = self.vae_decode(z).reshape([-1, 256, 1, 1])
            N, C2, H, W = x.shape
            vae_tmp = self.vae_decode(z).expand(-1, -1, H, W)
            #vae_tmp = x * z
            #vae_tmp += x
            
            vae_tmp = self.relu(vae_tmp)
            x = self.relu(x)
            # x = self.relu(vae_tmp)
        feat = self.DeepLab_decoder(low_level_features, vae_tmp, input['rgb'].cuda())
        feat = feat*0.1
        feat += self.DeepLab_decoder(low_level_features, x, input['rgb'].cuda())
        return {'pred': feat}

    def vae_encode(self, feat, feat_gt):

        feat = self.activation(self.fc_e0(feat))
        feat = self.activation(self.fc_e1(feat))

        feat_gt = self.activation(self.fc_e0(feat_gt))
        feat_gt = self.activation(self.fc_e1(feat_gt))

        if self.distribution == 'normal':
            z_mean = self.fc_mean(feat)
            z_var = F.softplus(self.fc_var(feat_gt))
        elif self.distribution == 'vmf':
            z_mean = self.fc_mean(feat)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            z_var = F.softplus(self.fc_var(feat_gt)) + 1
        else:
            raise NotImplemented

        return z_mean, z_var

    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            # p_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
            # p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        else:
            raise NotImplemented
        return q_z, p_z

    def vae_decode(self, z):
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)
        return x


if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True, _print=True)
    model.eval()
    # image = torch.randn(1, 3, 512, 1024)
    # with torch.no_grad():
    #     output = model.forward(image)
    # print(output.size())
    from torchsummary import summary

    summary(model, (3, 512, 1024), device='cpu')
