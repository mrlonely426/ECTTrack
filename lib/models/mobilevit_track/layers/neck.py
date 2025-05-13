import torch.nn as nn
import torch
from .connect_lighttrack import PWCA
from .lora_layer import MTLoRAConv, CompatConv, LoRACAModule


########## BN adjust layer before Correlation ##########
# borrowed from LightTrack (./lib/models/submodels.py)
class BN_adj(nn.Module):
    def __init__(self, num_channel):
        super(BN_adj, self).__init__()
        self.BN_x = nn.BatchNorm2d(num_channel)
        self.BN_z = nn.BatchNorm2d(num_channel)

    def forward(self, xf, zf):
        return self.BN_x(xf), self.BN_z(zf)


class Point_Neck_Mobile_simple_DP(nn.Module):
    def __init__(self, num_kernel_list=(256, 64), cat=False, matrix=True, adjust=True, adj_channel=128):
        super(Point_Neck_Mobile_simple_DP, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = torch.nn.ModuleList()
        self.adj_layer = torch.nn.ModuleList()
        for num_kernel in num_kernel_list:
            self.pw_corr.append(PWCA(num_kernel, cat=cat, CA=True, matrix=matrix))
            self.adj_layer.append(nn.Conv2d(num_kernel, adj_channel, 1))

    def forward(self, kernel, search, stride_idx):
        '''stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16'''
        oup = {}
        corr_feat = self.pw_corr[stride_idx]([kernel], [search])
        if self.adjust:
            corr_feat = self.adj_layer[stride_idx](corr_feat)
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup


'''Point-wise Correlation & channel adjust layer'''
class PW_Corr_adj(nn.Module):
    def __init__(self, cfg=None, num_kernel=64, cat=False, matrix=True, adj_channel=128):
        super(PW_Corr_adj, self).__init__()
        self.tasks = cfg.LORA.TASKS 
        self.pw_corr = PWCA(num_kernel, cat=cat, CA=True, matrix=matrix)
        if cat:
            num_kernel = num_kernel + 384
            mid_channel = 256
        else:
            mid_channel = 128
        if adj_channel is not None:
            if cfg.LORA.ENABLED:
                self.adj_layer = nn.Sequential(MTLoRAConv(num_kernel, mid_channel, kernel_size=1,use_norm=False, use_act=False,
                                                r=cfg.LORA.R_PER_TASK if cfg.LORA.ENABLED else 0,lora_shared_scale = cfg.LORA.SHARED_SCALE,
                                                lora_task_scale=cfg.LORA.TASK_SCALE,lora_dropout=cfg.LORA.DROPOUT,
                                                tasks=(cfg.LORA.TASKS if (cfg.LORA.ENABLED or cfg.LORA.INTERMEDIATE_SPECIALIZATION) else None),
                                                trainable_scale_shared=cfg.LORA.TRAINABLE_SCALE_SHARED,trainable_scale_per_task=cfg.LORA.TRAINABLE_SCALE_PER_TASK,
                                                shared_mode=cfg.LORA.SHARED_MODE),
                                                LoRACAModule(channels=mid_channel,lora_config=cfg.LORA),
                                                MTLoRAConv(mid_channel, adj_channel, kernel_size=1,use_norm=False, use_act=False,
                                                r=cfg.LORA.R_PER_TASK if cfg.LORA.ENABLED else 0,lora_shared_scale = cfg.LORA.SHARED_SCALE,
                                                lora_task_scale=cfg.LORA.TASK_SCALE,lora_dropout=cfg.LORA.DROPOUT,
                                                tasks=(cfg.LORA.TASKS if (cfg.LORA.ENABLED or cfg.LORA.INTERMEDIATE_SPECIALIZATION) else None),
                                                trainable_scale_shared=cfg.LORA.TRAINABLE_SCALE_SHARED,trainable_scale_per_task=cfg.LORA.TRAINABLE_SCALE_PER_TASK,
                                                shared_mode=cfg.LORA.SHARED_MODE),
                                                )
        else:
            self.adj_layer = nn.Sequential(CompatConv(in_channels=num_kernel, out_channels=mid_channel, kernel_size=1),
                                            CAModule(channels=mid_channel),
                                            CompatConv(in_channels=mid_channel, out_channels=adj_channel, kernel_size=1))
    def forward(self, kernel, search):
        '''stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16'''
        corr_feat = self.pw_corr([kernel], [search])
        x_tasks = {}
        for task in self.tasks:
            x_tasks[task] = corr_feat
        for adj in self.adj_layer:
            corr_feat,x_tasks = adj(corr_feat,x_tasks)
        return corr_feat, x_tasks


def build_subnet_feat_fusor(path_ops, model_cfg, cat=False, matrix=True, adj_channel=128):
    stride = model_cfg.strides[path_ops[0]]
    stride_idx = model_cfg.strides_use_new.index(stride)
    num_kernel = model_cfg.num_kernel_corr[stride_idx]
    return PW_Corr_adj(num_kernel=num_kernel, cat=cat, matrix=matrix, adj_channel=adj_channel)


def build_neck(cfg, hidden_dim):

    if cfg.MODEL.NECK.TYPE in ["BN_PWXCORR", "BN_SSAT", "BN_HSSAT"]:
        bn_adj = BN_adj(hidden_dim)
        return bn_adj
    else:
        raise ValueError("NECK TYPE %s is not supported." % cfg.MODEL.NECK.TYPE)


def build_feature_fusor(cfg, in_features, xcorr_out_features):

    if cfg.MODEL.NECK.TYPE == "BN_PWXCORR":

        if cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR != cfg.MODEL.HEAD.NUM_CHANNELS:
            adj_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        else:
            adj_channel = None
        pw_feature_fusor = PW_Corr_adj(cfg, num_kernel=xcorr_out_features, cat=True, adj_channel=adj_channel)

        return pw_feature_fusor

    else:
        raise ValueError("NECK TYPE %s is not supported." % cfg.MODEL.NECK.TYPE)