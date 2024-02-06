# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules
YOLO with decoupled detection head and segmentation head
add anchor free

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
import torch
from torch import tensor
import yaml  # for torch hub

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative


from zhuanhuanshuju1 import MultiHeadedAttention
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync
import torch.nn.functional as F

# hanfujun 
##########################DAN##################################
from dalib.modules.kernels import GaussianKernel
from  dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
##########################JAN##################################
from dalib.adaptation.jan import JointMultipleKernelMaximumMeanDiscrepancy
##########################DANN################################
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
###########################CDAN##################################
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
############################MDD##################################
from dalib.adaptation.mdd import MarginDisparityDiscrepancy

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


################################定义MMD损失#####################
import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据(n * len(x))
    target: 目标域数据(m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

######################CORAL算法###########################            
"""
Deep CORAL: Correlation Alignment for Deep Domain Adaptation, ECCV 2016, 很吃显存 需要80多个 这里没法运行
"""
def CORAL(source, target, **kwargs):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    """
    xm.t: 这是对张量 xm 进行转置操作
    @: 这是Python中的矩阵乘法运算符。@ 用于执行矩阵相乘操作，它将转置后的 xm 与原始的 xm 相乘。
    xm.t() @ xm: 这是对转置后的 xm 和原始 xm 进行矩阵相乘操作。结果将是一个新的矩阵，表示 xm 的转置矩阵与原始 xm 矩阵的乘积。
    / (ns - 1): 这是对先前的矩阵进行除法操作。ns 是一个变量，表示样本数量。通过 (ns - 1)，对矩阵中的每个元素进行除以 (ns - 1) 的操作。
    """
    xc = xm.t() @ xm / (ns - 1)
    
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)
    
    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss

######################BNM算法###########################            
"""
Towards Discriminability and Diversity:Batch Nuclear-norm Maximization under Label Insufficient Situations, CVPR 2020
"""  
import torch

def BNM(src, tar):
    """ Batch nuclear-norm maximization, CVPR 2020.
    tar: a tensor, softmax target output.
    NOTE: this does not require source domain data.
    """
    # print(type(tar))
    tar = tar.float() # 这里将tensor格式的tar转换成float格式的 如果直接使用原来的tensor格式torch会报错：RuntimeError: "svd_cuda_gesvdj" not implemented for 'Half'
    _, out, _ = torch.svd(tar)
    """
    注意这里loss前面有一个负号。我感觉有错误。这个符号会导致loss变成负数。loss = Loss + domain_loss 的值，从而导致结果变差
    因此, 我们这里把loss前面的负号修改为正号
    """
    # loss = -torch.mean(out) 
    loss = torch.mean(out)
    return loss

########################LMMDLoss########################
# from loss_funcs.mmd import MMDLoss 这个.py文件中就有
# from loss_funcs.adv import LambdaSheduler # 我直接复制过来了
# import torch # 上面有
# import numpy as np # 上面有


class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb
    
    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

class LMMDLoss(MMDLoss, LambdaSheduler):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, 
                    gamma=1.0, max_iter=1000, **kwargs):
        '''
        Local MMD
        '''
        super(LMMDLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
        super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
        self.num_class = num_class

    def forward(self, source, target, source_label, target_logits):
        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")
        
        elif self.kernel_type == 'rbf':
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
            weight_ss = torch.from_numpy(weight_ss).cuda() # B, B
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()

            kernels = self.guassian_kernel(source, target,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            loss = torch.Tensor([0]).cuda()
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]

            loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
            # Dynamic weighting
            lamb = self.lamb()
            self.step()
            loss = loss * lamb
            return loss
    
    def cal_weight(self, source_label, target_logits):
        batch_size = source_label.size()[0]
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label] # one hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum # label ratio

        # Pseudo label
        target_label = target_logits.cpu().data.max(1)[1].numpy()

        target_logits = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = target_logits / target_logits_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(source_label)
        set_t = set(target_label)
        count = 0
        for i in range(self.num_class): # (B, C)
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1) # (B, 1)
                t_tvec = target_logits[:, i].reshape(batch_size, -1) # (B, 1)
                
                ss = np.dot(s_tvec, s_tvec.T) # (B, B)
                weight_ss = weight_ss + ss
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st     
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

#####################################以下是原来yolodhs.py文件中自带的############################
class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80, width=1.0, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.merge = Conv(ch, 256 * width, 1, 1)
        self.cls_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.cls_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.cls_preds = nn.Conv2d(256 * width, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 * width, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 * width, 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out

"""
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
      

        return x if self.training else (torch.cat(z, 1), x)
"""


##原来的
class Detect(nn.Module):
    # anchor free with decoupled head
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.n_anchors = 1
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        #self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m = nn.ModuleList(DecoupledHead(x,nc,1,anchors) for x in ch) # hanfujun

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x (bs, 255, 20, 20) to x (bs , 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', seg_cfg="segheads.yaml", ch=3, nc=None, segnc=20, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
            self.yaml_file = Path(seg_cfg).name
            with open(seg_cfg, encoding='ascii', errors='ignore') as f:
                self.seg_yaml = yaml.safe_load(f)  # model dict

        else:  # is *.yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict
            with open(seg_cfg, encoding='ascii', errors='ignore') as f2:
                self.seg_yaml = yaml.safe_load(f2)  # model dict

        if len(self.yaml['backbone']) == 10:
            self.out_idx = [24, 33]  # model output index
            self.det_idx = 24
        #elif len(self.yaml['backbone']) == 10:
            #self.out_idx = [24, 33]  # model output index
            #self.det_idx = 24
        elif len(self.yaml['backbone']) == 12:
            self.out_idx = [33, 42]  # model output index
            self.det_idx = 33
        else:
            print("Only support yolov5* and yolov5*6!")
            assert (len(self.yaml['backbone']) in [10,12])
        self.segnc = segnc


        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # 设置类别数，不过一般不执行，因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # 重写anchor，一般不执行，因为传进来的anchors一般都是None
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 创建网络模型
        # self.model: 初始化的整个网络模型（包括Detect层结构）
        # self.save: 所有层结构中from不等于-1的序号，并排好序 [4,6,10,14,17,20,23]
        self.model, self.save = parse_model(deepcopy(self.yaml), deepcopy(self.seg_yaml), ch=[ch])  # model, savelist
        # default class name ['0','1','2','3'.....'19']
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        # self.inplace = True 默认使用True 不使用加速推理
        self.inplace = self.yaml.get('inplace', True)
        # LOGGER.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        ### hanfujun 领域自适应
        ################################## DAN ####################
        # kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        # self.loss = MultipleKernelMaximumMeanDiscrepancy(kernels)

        ################################## JAN ####################
        # layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
        # layer2_kernels = (GaussianKernel(1.),)
        # self.loss = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, layer2_kernels))
        ################################## DANN ####################
        """
        注意这里的in_feature的维度, 这个和要适配的那个特征层有关系。比如说，
        用5s模型的话, 适配第9层。由于5s第9层的维度大小为20 x 20 x 512 = 204800
        同理, 如果是5l第9层的维度的大小为 20 x 20 x 1024 = 409600 依次类推. 所以这里的in_feature要根据网络模型和大小来定
        """
        discriminator1 = DomainDiscriminator(in_feature=819200, hidden_size=256) #128
        self.loss1 = DomainAdversarialLoss(discriminator1, reduction='mean')

        discriminator2 = DomainDiscriminator(in_feature=409600, hidden_size=256)
        self.loss2 = DomainAdversarialLoss(discriminator2, reduction='mean')

        discriminator3 = DomainDiscriminator(in_feature=204800, hidden_size=256)
        self.loss3 = DomainAdversarialLoss(discriminator3, reduction='mean')

        ################################## 多头注意力机制--DANN ####################
        # """
        # 注意这里的in_feature的维度, 这个和要适配的那个特征层有关系。比如说，
        # 用5s模型的话, 适配第9层。由于5s第9层的维度大小为20 x 20 x 512 = 204800
        # 同理, 如果是5l第9层的维度的大小为 20 x 20 x 1024 = 409600 依次类推. 所以这里的in_feature要根据网络模型和大小来定
        # """
        # discriminator1 = DomainDiscriminator(in_feature=6400, hidden_size=256) #128
        # self.loss1 = DomainAdversarialLoss(discriminator1, reduction='mean')

        # discriminator2 = DomainDiscriminator(in_feature=2048, hidden_size=256)
        # self.loss2 = DomainAdversarialLoss(discriminator2, reduction='mean')

        # discriminator3 = DomainDiscriminator(in_feature=1024, hidden_size=256)
        # self.loss3 = DomainAdversarialLoss(discriminator3, reduction='mean')

        ################################### CDAN #####################
        """
        基础的是5s第9层 20 x 20 x 512 = 204800  但是这里的in_feature = 204800 * num_classes
        """
        #discriminator = DomainDiscriminator(in_feature=204800, hidden_size=1024)
        #self.loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')

        ######################MDD#########################
        # self.loss = MarginDisparityDiscrepancy(margin=4.)

        # Build strides, anchors
        # 获取Detect模块的stride（相对输入图像的下采样率）和anchors在当前Detect输出的feature map的尺度
        # m = self.model[24]  # Detect()  
        # m = self.model[-1]  # Detect() # hanfujun
        # 源代码当中这里的 m = self.model[-1] 而这里我们使用的是self.det_idx
        # 这是因为我们这里使用的检测和分割的模式，如果取-1的话，并不是原始的代表-1的detect的三个层 而是最后的分割层 
        # 所以这里需要将最后的-1修改成这个det_idx层  因为这里的det_idx就等于原始代码中的-1层
        m = self.model[self.det_idx] # Detect()
        #print(m) # 检查过
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            r1, _ = self.forward(torch.zeros(1, ch, s, s), isdomain=False) # hanfujun
            #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', len(r1), _)
            """
            这里做一个调试bug的记录
            这里获取的是m.stride的下采样倍率。注意这里的x是从r1中获取
            而这里的r1获取的地方是sefl.forward这个函数中获取
            然后进入到self.forward这个函数中可以发现
            return self._forward_once(x, domainx, profile, visualize, isdomain) 返回的是self._forward_once的输出
            然后进入到self._forward_once函数中, 返回的是output 和 d1+d2+d3 而原来只有output.
            所以需要把后面的 d1+d2+d3 给注释掉。 但是如果需要的话,后续还是需要继续修改的
            """
            m.stride = torch.tensor([s / x.shape[-2] for x in r1[0]]) # hanfujun

            # m.stride = torch.tensor([ 8., 16., 32.])
            # m.stride = np.array(m.stride)
            # m.stride = torch.tensor(self.det_stride)
            # print('output11:', output)
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0]])  # forward # 原来的

            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            #self._initialize_biases()  # only run once # hanfujun

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, domainx=None, augment=False, profile=False, visualize=False, isdomain=True):
        if augment:
            return self._forward_augment(x,isdomain)  # augmented inference, None
        #print('mmmmmmmmmmmmmmmmmmmmmmmmmm', x)
        #print('nnnnnnnnnnnnnnnnnnnnnnnnnnn', domainx)
        return self._forward_once(x, domainx, profile, visualize, isdomain)  # single-scale inference, train #hanfujun

    def _forward_augment(self, x, isdomain=True):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi, isdomain=isdomain)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train
    

    def _forward_once(self, x, domainx, profile=False, visualize=False, isdomain=True):
        y, dt = [], []  # outputs
        y1=[]
        d1=0
        d2=0
        d3=0
        #print('xxxxxxxxxxxxxxxx:',x.shape)
        #img1 = torch.rand(x.shape).to(x.device)
        x1 = domainx
        #print('x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1:',x1)
        # print('xxxx')
        # if(domainx is None):
        #    print("forword-----is none")
        # else:
        #    print("forword-----", x.shape," ",x1.shape, " ",domainx.shape)
        output = []
        #print(x.shape[0])
        if (x.shape[0]==1):
            isdomain=False
        #print("kkkkkk:",self.model)
        # if (x.shape[0]!=1):
        #     isdomain=True
        for idx, m in enumerate(self.model):
            #print("the ",idx," th module:",m)
            #print('ssssssssssssssssss:', m.f)
            if m.f != -1:  # if not from previous layer
                """
                这里需要做4个concat操作和1个Detect操作
                concat操作如m.f = [-1, 6] x 就有两个元素 一个是上一层的输出,另一个是idx=6的层的输出 在送到x=m(x)的concat操作
                Detect操作m.f = [17, 20, 23] x有三个元素, 分别是存放第17层和第20层第23层输出  再送到x=m(x) 做Detect的forward
                """
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                #print('oooooooo:', x)
                #print('hanhanhan1111111:', x.shape)
                if(isdomain):
                    #print('llllllllllllllllllllllllllllllllll')
                    x1 = y1[m.f] if isinstance(m.f, int) else [x1 if j == -1 else y[j] for j in m.f]
                    #print('hanhanhan1111111:', x1.shape)
            if profile:
                self._profile_one_layer(m, x, dt) # 原来的 hanfujun
                # c = isinstance(m, Detect)  # copy input as inplace fix
                # o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                # t = time_sync()
                # for _ in range(10):
                #     m(x.copy() if c else x)
                # dt.append((time_sync() - t) * 100)
                # if m == self.model[0]:
                #     LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                # LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
            #print('mmmmmmmmmmmmmm', m)
            x = m(x)  # run
            #"""
            if (idx < 24 and isdomain): #15
                #print('hanhanhan:', isdomain)
                #print('@@@@@@@@@@@@@@@@@@@@@:', x1)
                x1 = m(x1) #m(x1)
                # if (idx == 24):
                #     print("layer", idx, len(x), len(x1)) 
                # if (idx==4): 
                #     pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                #     output_tensor_x = pooling_layer(pooling_layer(pooling_layer(pooling_layer(x))))
                #     output_tensor_x1 = pooling_layer(pooling_layer(pooling_layer(pooling_layer(x1))))
                    
                #     d_model = 2  # 输入序列的维度
                #     seq_len_1 = output_tensor_x.shape[1] * output_tensor_x.shape[2] * output_tensor_x.shape[3] # 输入序列的长度 (d_model,seq_len)的维度是 d_model (列) x seq_len (行)
                #     seq_len_2 = output_tensor_x1.shape[1] * output_tensor_x1.shape[2] * output_tensor_x1.shape[3]
                #     batch_size = output_tensor_x.shape[0]
                #     num_heads = 2
                    
                #     shape1 = (batch_size, seq_len_1, seq_len_1)
                #     mask1 = torch.from_numpy(np.random.rand(*shape1).astype(np.float64)) # 这里的mask的维度应该是(seq_len,seq_len): seq_len x seq_len
                #     shape2 = (batch_size, seq_len_2, seq_len_2)
                #     mask2 = torch.from_numpy(np.random.rand(*shape2).astype(np.float64))
                #     input1 = torch.rand(batch_size, seq_len_1, d_model) # batch_size x seq_len x d_model
                #     input2 = torch.rand(batch_size, seq_len_2, d_model) # batch_size x seq_len x d_model
                #     multi_attn = MultiHeadedAttention(num_heads = num_heads, d_model = d_model, dropout = 0.1)
                #     out1 = multi_attn(query = input1, key = input1, value = input1, mask = mask1)
                #     out2 = multi_attn(query = input2, key = input2, value = input2, mask = mask2)

                #     # DAN
                #     # batch_size = out1.shape[0]
                #     # feature_size = out1.shape[1] * out1.shape[2]
                #     # t1 = out1.reshape(batch_size, feature_size)
                #     # t2 = out2.reshape(batch_size, feature_size)
                #     # d1 = self.loss(t1, t2)
                #     device = 'cuda:0'
                #     batch_size = out1.shape[0]
                #     feature_size = out1.shape[1] * out1.shape[2]
                #     t1s = (out1.reshape(batch_size, feature_size)).to(device)
                #     t1t = (out2.reshape(batch_size, feature_size)).to(device)
                #     d1 = self.loss1(t1s,t1t)
                """
                if (idx==6): 
                    pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                    output_tensor_x = pooling_layer(pooling_layer(pooling_layer(pooling_layer(x))))
                    output_tensor_x1 = pooling_layer(pooling_layer(pooling_layer(pooling_layer(x1))))
                    
                    d_model = 2  # 输入序列的维度
                    seq_len_1 = output_tensor_x.shape[1] * output_tensor_x.shape[2] * output_tensor_x.shape[3] # 输入序列的长度 (d_model,seq_len)的维度是 d_model (列) x seq_len (行)
                    seq_len_2 = output_tensor_x1.shape[1] * output_tensor_x1.shape[2] * output_tensor_x1.shape[3]
                    batch_size = output_tensor_x.shape[0]
                    num_heads = 2
                    
                    shape1 = (batch_size, seq_len_1, seq_len_1)
                    mask1 = torch.from_numpy(np.random.rand(*shape1).astype(np.float64)) # 这里的mask的维度应该是(seq_len,seq_len): seq_len x seq_len
                    shape2 = (batch_size, seq_len_2, seq_len_2)
                    mask2 = torch.from_numpy(np.random.rand(*shape2).astype(np.float64))
                    input1 = torch.rand(batch_size, seq_len_1, d_model) # batch_size x seq_len x d_model
                    input2 = torch.rand(batch_size, seq_len_2, d_model) # batch_size x seq_len x d_model
                    multi_attn = MultiHeadedAttention(num_heads = num_heads, d_model = d_model, dropout = 0.1)
                    out1 = multi_attn(query = input1, key = input1, value = input1, mask = mask1)
                    out2 = multi_attn(query = input2, key = input2, value = input2, mask = mask2)

                    # DAN
                    # batch_size = out1.shape[0]
                    # feature_size = out1.shape[1] * out1.shape[2]
                    # t1 = out1.reshape(batch_size, feature_size)
                    # t2 = out2.reshape(batch_size, feature_size)
                    # d1 = self.loss(t1, t2)
                    device = 'cuda:0'
                    batch_size = out1.shape[0]
                    feature_size = out1.shape[1] * out1.shape[2]
                    t1s = (out1.reshape(batch_size, feature_size)).to(device)
                    t1t = (out2.reshape(batch_size, feature_size)).to(device)
                    d2 = self.loss2(t1s,t1t)


                if (idx==8): 
                    pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                    output_tensor_x = pooling_layer(pooling_layer(pooling_layer(pooling_layer(x))))
                    output_tensor_x1 = pooling_layer(pooling_layer(pooling_layer(pooling_layer(x1))))
                    
                    d_model = 2  # 输入序列的维度
                    seq_len_1 = output_tensor_x.shape[1] * output_tensor_x.shape[2] * output_tensor_x.shape[3] # 输入序列的长度 (d_model,seq_len)的维度是 d_model (列) x seq_len (行)
                    seq_len_2 = output_tensor_x1.shape[1] * output_tensor_x1.shape[2] * output_tensor_x1.shape[3]
                    batch_size = output_tensor_x.shape[0]
                    num_heads = 2
                    
                    shape1 = (batch_size, seq_len_1, seq_len_1)
                    mask1 = torch.from_numpy(np.random.rand(*shape1).astype(np.float64)) # 这里的mask的维度应该是(seq_len,seq_len): seq_len x seq_len
                    shape2 = (batch_size, seq_len_2, seq_len_2)
                    mask2 = torch.from_numpy(np.random.rand(*shape2).astype(np.float64))
                    input1 = torch.rand(batch_size, seq_len_1, d_model) # batch_size x seq_len x d_model
                    input2 = torch.rand(batch_size, seq_len_2, d_model) # batch_size x seq_len x d_model
                    multi_attn = MultiHeadedAttention(num_heads = num_heads, d_model = d_model, dropout = 0.1)
                    out1 = multi_attn(query = input1, key = input1, value = input1, mask = mask1)
                    out2 = multi_attn(query = input2, key = input2, value = input2, mask = mask2)

                    # DAN
                    # batch_size = out1.shape[0]
                    # feature_size = out1.shape[1] * out1.shape[2]
                    # t1 = out1.reshape(batch_size, feature_size)
                    # t2 = out2.reshape(batch_size, feature_size)
                    # d1 = self.loss(t1, t2)
                    device = 'cuda:0'
                    batch_size = out1.shape[0]
                    feature_size = out1.shape[1] * out1.shape[2]
                    t1s = (out1.reshape(batch_size, feature_size)).to(device)
                    t1t = (out2.reshape(batch_size, feature_size)).to(device)
                    d3 = self.loss3(t1s,t1t)


                # if (idx==4):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d1 = self.loss1(t1s,t1t)
                """
                ############################DAN, 已经运行成功, 注意修改原来的修改functionnal.py中的一些配置#####################
                '''
                Learning Transferable Features with Deep Adaptation Networks 2015  ICML
                原来的MMD:source和target用一个相同的映射映射在一个再生核希尔伯特空间(RKHS)中，然后求映射后两部分数据的均值差异，就当作是两部分数据的差异
                '''
                # if (idx == 5):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1 = x.reshape(batch_size, feature_size)
                #     t2 = x1.reshape(batch_size, feature_size)
                #     d1 = self.loss(t1, t2)
                # if (idx == 7):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1 = x.reshape(batch_size, feature_size)
                #     t2 = x1.reshape(batch_size, feature_size)
                #     d2 = self.loss(t1, t2)
                # if (idx == 9):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1 = x.reshape(batch_size, feature_size)
                #     t2 = x1.reshape(batch_size, feature_size)
                #     d3 = self.loss(t1, t2)
                ############################JAN, 已经运行成功, 注意修改原来的修改functionnal.py中的一些配置#####################
                '''
                Joint Adaptation Network ICML 2017
                '''
                # if (idx==8):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t8s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t8t = x1.reshape(batch_size, feature_size)
                # if (idx==9):
                #     batch_size = x.shape[0] # batch_size
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3] # gai ceng weidu
                #     t9s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t9t = x1.reshape(batch_size, feature_size)
                #     #t2 = x1.reshape(batch_size,feature_size) # reshape cheng xiang ying ge shi  mu biao yu
                #     d1 = self.loss((t8s,t9s),(t8t,t9t))

                # if (idx==8):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t8s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t8t = x1.reshape(batch_size, feature_size)
                # if (idx==9):
                #     batch_size = x.shape[0] # batch_size
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3] # gai ceng weidu
                #     t9s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t9t = x1.reshape(batch_size, feature_size)
                #     #t2 = x1.reshape(batch_size,feature_size) # reshape cheng xiang ying ge shi  mu biao yu
                #     d2 = self.loss((t8s,t9s),(t8t,t9t))

                # if (idx==22):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t5s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t5t = x1.reshape(batch_size, feature_size)
                # if (idx==23):
                #     batch_size = x.shape[0] # batch_size
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3] # gai ceng weidu
                #     t6s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t6t = x1.reshape(batch_size, feature_size)
                #     #t2 = x1.reshape(batch_size,feature_size) # reshape cheng xiang ying ge shi  mu biao yu
                #     d3 = self.loss((t5s,t6s),(t5t,t6t))


                #########################DANN, 已经运行成功。但是要修改functionnal.py中的一些配置。因为原本binary_cross_entropy和binary_cross_entropy_with_logits不匹配###############
                #不匹配的原因主要是因为返回有logits。这个返回的logits是在DomainDiscriminator函数中返回的。具体的可以去看https://dalib.readthedocs.io/en/latest/dalib.adaptation.html#id6
                '''
                Domain-Adversarial Training of Neural Networks JMLR 2016
                '''
                # if (idx==4):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d1 = self.loss1(t1s,t1t)
                    
                # if (idx==6):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d2 = self.loss2(t1s,t1t)

                # if (idx==8):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d3 = self.loss3(t1s,t1t)
                #########################DANN, 已经运行成功。但是要修改functionnal.py中的一些配置。因为原本binary_cross_entropy和binary_cross_entropy_with_logits不匹配###############
                #不匹配的原因主要是因为返回有logits。这个返回的logits是在DomainDiscriminator函数中返回的。具体的可以去看https://dalib.readthedocs.io/en/latest/dalib.adaptation.html#id6
                '''
                Domain-Adversarial Training of Neural Networks JMLR 2016
                '''

                # if (idx==4):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d1 = self.loss1(t1s,t1t)
                #     # 先进行梯度反转
                #     from dalib.modules.grl import WarmStartGradientReverseLayer
                #     self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
                #     f = self.grl(torch.cat((t1s, t1t), dim=0))
                #     t1s_ad, pre_t_ad = f.chunk(2, dim=0) 
                #     print(t1s_ad.shape[0])
                #     #print(t1s_ad)
                #     # 执行你的代码
                #     pre_s, pre_t = F.softmax(t1s_ad, dim=1), F.softmax(pre_t_ad, dim=1)
                #     # 直接计算核范数
                #     loss1 = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / t1s_ad.shape[0]
                #     d1 = d1 + loss1
                
                # if (idx==6):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d2 = self.loss2(t1s,t1t)
                #     # 先进行梯度反转
                #     from dalib.modules.grl import WarmStartGradientReverseLayer
                #     self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
                #     f = self.grl(torch.cat((t1s, t1t), dim=0))
                #     t1s_ad, pre_t_ad = f.chunk(2, dim=0) 
                #     #print(t1s_ad)
                #     # 执行你的代码
                #     pre_s, pre_t = F.softmax(t1s_ad, dim=1), F.softmax(pre_t_ad, dim=1)
                #     # 直接计算核范数
                #     loss2 = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / t1s_ad.shape[0]
                #     d2 = d2 + loss2

                # if (idx==8):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d3 = self.loss3(t1s,t1t) 
                #     # 先进行梯度反转
                #     from dalib.modules.grl import WarmStartGradientReverseLayer
                #     self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
                #     f = self.grl(torch.cat((t1s, t1t), dim=0))
                #     t1s_ad, pre_t_ad = f.chunk(2, dim=0) 
                #     #print(t1s_ad)
                #     # 执行你的代码
                #     pre_s, pre_t = F.softmax(t1s_ad, dim=1), F.softmax(pre_t_ad, dim=1)
                #     # 直接计算核范数
                #     loss3 = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / t1s_ad.shape[0]
                #     d3 = d3 + loss3  


                # 高斯核函数
                # if (idx==4):
                #     # 示例数据
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d1 = self.loss1(t1s,t1t) 
                #     # 先进行梯度反转
                #     from dalib.modules.grl import WarmStartGradientReverseLayer
                #     self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
                #     f = self.grl(torch.cat((t1s, t1t), dim=0))
                #     t1s_ad, pre_t_ad = f.chunk(2, dim=0)
                #     # 执行你的代码
                #     pre_s, pre_t = F.softmax(t1s_ad, dim=1), F.softmax(pre_t_ad, dim=1)
                #     # 定义高斯核函数
                #     def gaussian_kernel(x, y, sigma=1.0):
                #         return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))
                #     # 计算高斯核范数
                #     def compute_gaussian_kernel_norm(x, y, kernel_func, sigma=1.0):
                #         norm_sum = 0.0
                #         for i in range(x.size(0)):
                #             for j in range(y.size(0)):
                #                 norm_sum += kernel_func(x[i], y[j], sigma)
                #         return norm_sum
                #     # 使用高斯核计算核范数
                #     sigma_value = 3  # 调整高斯核的参数
                #     gaussian_norm = compute_gaussian_kernel_norm(pre_s, pre_t, gaussian_kernel, sigma=sigma_value)
                #     # 计算损失
                #     loss1 = (-gaussian_norm) / pre_s.shape[0]
                #     d1 = d1 + loss1 

                # if (idx==6):
                #     # 示例数据
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d2 = self.loss2(t1s,t1t) 
                #     # 先进行梯度反转
                #     from dalib.modules.grl import WarmStartGradientReverseLayer
                #     self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
                #     f = self.grl(torch.cat((t1s, t1t), dim=0))
                #     t1s_ad, pre_t_ad = f.chunk(2, dim=0)
                #     # 执行你的代码
                #     pre_s, pre_t = F.softmax(t1s_ad, dim=1), F.softmax(pre_t_ad, dim=1)
                #     # 定义高斯核函数
                #     def gaussian_kernel(x, y, sigma=1.0):
                #         return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))
                #     # 计算高斯核范数
                #     def compute_gaussian_kernel_norm(x, y, kernel_func, sigma=1.0):
                #         norm_sum = 0.0
                #         for i in range(x.size(0)):
                #             for j in range(y.size(0)):
                #                 norm_sum += kernel_func(x[i], y[j], sigma)
                #         return norm_sum
                #     # 使用高斯核计算核范数
                #     sigma_value = 3  # 调整高斯核的参数
                #     gaussian_norm = compute_gaussian_kernel_norm(pre_s, pre_t, gaussian_kernel, sigma=sigma_value)
                #     # 计算损失
                #     loss2 = (-gaussian_norm) / pre_s.shape[0]
                #     d2 = d2 + loss2 

                # if (idx==8):
                #     # 示例数据
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                #     t1t = x1.reshape(batch_size, feature_size)
                #     d3 = self.loss3(t1s,t1t) 
                #     # 先进行梯度反转
                #     from dalib.modules.grl import WarmStartGradientReverseLayer
                #     self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
                #     f = self.grl(torch.cat((t1s, t1t), dim=0))
                #     t1s_ad, pre_t_ad = f.chunk(2, dim=0)
                #     # 执行你的代码
                #     pre_s, pre_t = F.softmax(t1s_ad, dim=1), F.softmax(pre_t_ad, dim=1)
                #     # 定义高斯核函数
                #     def gaussian_kernel(x, y, sigma=1.0):
                #         return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))
                #     # 计算高斯核范数
                #     def compute_gaussian_kernel_norm(x, y, kernel_func, sigma=1.0):
                #         norm_sum = 0.0
                #         for i in range(x.size(0)):
                #             for j in range(y.size(0)):
                #                 norm_sum += kernel_func(x[i], y[j], sigma)
                #         return norm_sum
                #     # 使用高斯核计算核范数
                #     sigma_value = 3  # 调整高斯核的参数
                #     gaussian_norm = compute_gaussian_kernel_norm(pre_s, pre_t, gaussian_kernel, sigma=sigma_value)
                #     # 计算损失
                #     loss3 = (-gaussian_norm) / pre_s.shape[0]
                #     d3 = d3 + loss3 
                #####################################################################################I-DANN#################################
                def iterative_hard_thresholding(matrix, num_iterations, threshold):
                    # 实际的迭代硬阈值算法需要根据阈值进行逐步优化
                    for _ in range(num_iterations):
                        matrix = torch.sign(matrix) * torch.clamp(torch.abs(matrix) - threshold, min=0.0)
                    return matrix

                if (idx==4):
                    #print('xxxxxxxxxxxxx:', x.shape)
                    batch_size = x.shape[0]
                    feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                    t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                    #print('xxxxxxxxxxxxx:', t1s.shape)
                    t1t = x1.reshape(batch_size, feature_size)
                    #print('xxxxxxxxxxxxx:', t1t.shape)
                    d1 = self.loss1(t1s,t1t) 
                    # 先进行梯度反转
                    from dalib.modules.grl import WarmStartGradientReverseLayer
                    self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
                    f = self.grl(torch.cat((t1s, t1t), dim=0))
                    t1s_ad, pre_t_ad = f.chunk(2, dim=0)
                    pre_s, pre_t = F.softmax(t1s_ad, dim=1), F.softmax(pre_t_ad, dim=1) 
                    # 迭代硬阈值算法近似核范数
                    # Normal-CMC->Foggy-CMC: 0.0001，迭代10; Normal-CMC->BDD-intrusion: 0.1，迭代20; Normal-CMC->Rainy-CMC: 0.1，迭代10 ; Normal-CMC->Night-CMC:0.001 迭代10
                    thresholded_pre_s = iterative_hard_thresholding(pre_s, num_iterations=10, threshold=0.001) 
                    thresholded_pre_t = iterative_hard_thresholding(pre_t, num_iterations=10, threshold=0.001)
                    # 直接计算核范数
                    loss1 = (-torch.norm(thresholded_pre_t, 'nuc') + torch.norm(thresholded_pre_s, 'nuc')) / t1s_ad.shape[0]
                    d1 = d1 + loss1

                if (idx==6):
                    batch_size = x.shape[0]
                    feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                    t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                    t1t = x1.reshape(batch_size, feature_size)
                    d2 = self.loss2(t1s,t1t) 
                    # 先进行梯度反转
                    from dalib.modules.grl import WarmStartGradientReverseLayer
                    self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
                    f = self.grl(torch.cat((t1s, t1t), dim=0))
                    t1s_ad, pre_t_ad = f.chunk(2, dim=0) 
                    pre_s, pre_t = F.softmax(t1s_ad, dim=1), F.softmax(pre_t_ad, dim=1) 
                    # 迭代硬阈值算法近似核范数
                    thresholded_pre_s = iterative_hard_thresholding(pre_s, num_iterations=10, threshold=0.001)#0.0001
                    thresholded_pre_t = iterative_hard_thresholding(pre_t, num_iterations=10, threshold=0.001)
                    # 直接计算核范数
                    loss2 = (-torch.norm(thresholded_pre_t, 'nuc') + torch.norm(thresholded_pre_s, 'nuc')) / t1s_ad.shape[0]
                    d2 = d2 + loss2

                if (idx==8):
                    batch_size = x.shape[0]
                    feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                    t1s = x.reshape(batch_size, feature_size)  # reshape cheng xiang ying ge shi
                    t1t = x1.reshape(batch_size, feature_size)
                    d3 = self.loss3(t1s,t1t) 
                    # 先进行梯度反转
                    from dalib.modules.grl import WarmStartGradientReverseLayer
                    self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
                    f = self.grl(torch.cat((t1s, t1t), dim=0))
                    t1s_ad, pre_t_ad = f.chunk(2, dim=0) 
                    pre_s, pre_t = F.softmax(t1s_ad, dim=1), F.softmax(pre_t_ad, dim=1) 
                    # 迭代硬阈值算法近似核范数
                    thresholded_pre_s = iterative_hard_thresholding(pre_s, num_iterations=10, threshold=0.001)
                    thresholded_pre_t = iterative_hard_thresholding(pre_t, num_iterations=10, threshold=0.001)
                    # 直接计算核范数
                    loss3 = (-torch.norm(thresholded_pre_t, 'nuc') + torch.norm(thresholded_pre_s, 'nuc')) / t1s_ad.shape[0]
                    d3 = d3 + loss3
                ############################# CDAN ################################
                #AttributeError: Can't pickle local object 'ConditionalDomainAdversarialLoss.__init__.<locals>.<lambda>'#
                # if (idx==9):
                #     num_classes = 1
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     f_s = x.reshape(batch_size, feature_size) # fs
                #     f_t = x1.reshape(batch_size,feature_size) # ft
                #     device = 'cuda:0'
                #     g_s = torch.randn((batch_size,num_classes), dtype=torch.float16).to(device)
                #     g_t = torch.randn((batch_size, num_classes), dtype=torch.float16).to(device)
                #     d1 = self.loss(g_s, f_s, g_t, f_t)
                ###############################MDD##################################
                # if (idx==9):
                #     num_classes = 4
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     device = 'cuda:0'
                #     y_s = torch.randn((batch_size, num_classes), dtype=torch.float16).to(device)
                #     y_t = torch.randn((batch_size, num_classes), dtype=torch.float16).to(device)
                #     y_s_adv = torch.randn((batch_size, num_classes), dtype=torch.float16).to(device)
                #     y_t_adv = torch.randn((batch_size, num_classes), dtype=torch.float16).to(device)
                #     d1 = self.loss(y_s,y_s_adv,y_t,y_t_adv)
                #################################### MMD,已经运行成功。模型的定义在最上面 ##################################
                """
                Deep Domain Confusion: Maximizing for Domain Invariance 2014 arXiv
                值得注意的是,MMD可以决定选择哪个特征层(“深度”)和适应层应该有多大(“宽度”)，是整个目标的关键部分。
                即网络中的适配层是哪一层和维度都是可以任意选择的,选择方法就是算MMD哪个最小用哪个,
                如上图图中,作者最后选了放在fc7后面,并且维度也会根据MMD的计算结果来微调。
                """
                # if (idx==5):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     f_s = x.reshape(batch_size, feature_size)
                #     f_t = x1.reshape(batch_size, feature_size)
                #     MMD = MMDLoss()
                #     d1 = MMD(f_s, f_t)
                # if (idx==7):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     f_s = x.reshape(batch_size, feature_size)
                #     f_t = x1.reshape(batch_size, feature_size)
                #     MMD = MMDLoss()
                #     d2 = MMD(f_s, f_t)
                # if (idx==9):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     f_s = x.reshape(batch_size, feature_size)
                #     f_t = x1.reshape(batch_size, feature_size)
                #     MMD = MMDLoss()
                #     d3 = MMD(f_s, f_t)
                ################################## BNM, 运行成功。模型的定义在最上面 ########################################
                # if (idx==9):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     f_s = x.reshape(batch_size, feature_size)
                #     f_t = x1.reshape(batch_size, feature_size)
                #     d1 = BNM(f_s, f_t)
                ############################## CORAL ##################################
                # if (idx==9):
                #     batch_size = x.shape[0]
                #     feature_size = x.shape[1] * x.shape[2] * x.shape[3]
                #     f_s = x.reshape(batch_size, feature_size)
                #     f_t = x1.reshape(batch_size, feature_size)
                #     d1 = CORAL(f_s, f_t)
                ########################################################################
            #"""
            #hanfujun 
            if idx in self.out_idx:
                output.append(x)
            # if idx == 24:
            #    print('idx:', idx, len(output))
            #    output.append(x)
            # if idx == 33:
            #    print('idx:', idx, len(output))
            #    output.append(x)
            y.append(x if m.i in self.save else None)  # save output
            #if(idx <= 24 and isdomain):
            if(isdomain):
                y1.append(x1 if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        # 这里返回的是两个变量
        #return output, d1+d2+d3
        
        if profile:
            LOGGER.info('%.1fms total' % sum(dt))
        
        # print('y1:', len(y1))
        # print('y:', len(y))
        # print('output:', len(output))
        # print('xxxx:', d1)
        
        #return output, d1+d2+d3 # d1+d2+d3
        # 可以考虑一下实例，比如小目标的权重等于小目标个数/总的实例个数 + 中目标的权重等于中目标个数/中目标总的实例个数 + 大目标的权重等于大目标个数/大总的实例个数
        # w1 = Σsmall/Σtotal 依次类推
        return output, d1+d2+d3 # d1+d2+d3 # 可以考虑一下实例，比如小目标的权重等于小目标个数/总的实例个数 +

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, sd,ch):  # model_dict, segmentation dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head'] + sd['SegHead']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is nn.Conv2d:
            args[0] = int(args[0]*gw)
            args[1] = int(sd['segnc'])
            c2 = ch[f]
        else:
            c2 = ch[f]

        
        # 原来的
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
    
    """
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
    """


# 原来的
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5l.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5l.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    """
