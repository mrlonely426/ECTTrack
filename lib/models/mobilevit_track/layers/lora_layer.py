import torch.nn as nn
import torch
from torch.nn import functional as F

from easydict import EasyDict as edict
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping

from .conv_layer import  ConvLayer
from .normalization_layers import get_normalization_layer
from .non_linear_layers import get_activation_fn

class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False


class LoRAConv(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        # ↓ the remaining part is for LoRA
        r: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        tasks=None,
        **kwargs,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.conv = nn.Conv2d(
            in_channels, out_channels,kernel_size, **kwargs)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_channels//self.conv.groups * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            # Merge the weights and mark it
            self.conv.weight.data += (self.lora_B @
                                        self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.conv(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = F.conv2d(x, (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling, self.conv.bias)  
        return pretrained + lora


class MTLoRAConv(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1 ,
        dilation: int = 1,
        groups: int = 1 ,
        bias: bool = True,
        use_norm: bool = False,
        use_act: bool = False,
        auto_padding: bool = True,
        # ↓ the remaining part is for LoRA
        r: Union[int, Mapping[str, int]] = 1,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = 0.0,
        tasks=None,
        trainable_scale_shared=False,
        trainable_scale_per_task=False,
        shared_mode: str = 'matrix',
        **kwargs,
    ):
        assert shared_mode in ['matrix', 'matrixv2',
                               'add', 'addition', 'lora_only']
        if shared_mode == 'add':
            shared_mode = 'addition'
        if shared_mode == 'lora_only':
            tasks = None
        has_tasks = tasks is not None
        if not has_tasks:
            if shared_mode not in ['matrix']:
                shared_mode = 'matrix'

        if isinstance(r, int):
            r = {'shared': r}
        super().__init__(
            r=r['shared'],lora_alpha=lora_shared_scale, lora_dropout=lora_dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_norm = use_norm
        self.use_act = use_act
        self.groups = groups
        self.bias = bias
        if auto_padding:
            self.padding = int((kernel_size - 1) / 2) * dilation
        else:
            self.padding = 0

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, groups=groups, padding=self.padding, bias=self.bias,**kwargs)
        if use_norm:
                self.norm_layer = nn.BatchNorm2d(out_channels)


        self.tasks = tasks
        self.shared_mode = shared_mode
        if r['shared'] > 0:
            if has_tasks:
                self.lora_tasks_A = nn.ParameterDict({
                    task: nn.Parameter(
                        self.conv.weight.new_zeros((r[task]* kernel_size, in_channels* kernel_size)))
                    for task in tasks
                })
                self.lora_tasks_B = nn.ParameterDict({
                    task: nn.Parameter(
                        self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r[task]* kernel_size)))
                    for task in tasks
                })
                if trainable_scale_per_task:
                    self.lora_task_scale = nn.ParameterDict({
                        task: nn.Parameter(torch.FloatTensor(
                            [lora_task_scale]))
                        for task in tasks
                    })
                else:
                    self.lora_task_scale = {task: lora_task_scale[task]
                                            for task in tasks}
            if self.shared_mode == 'addition':
                assert has_tasks
                self.lora_norm = nn.BatchNorm2d(out_channels)
            elif self.shared_mode == 'matrix' or self.shared_mode == 'matrixv2':
                self.lora_shared_A = nn.Parameter(
                    self.conv.weight.new_zeros((r['shared']* kernel_size, in_channels* kernel_size)))
                self.lora_shared_B = nn.Parameter(
                    self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r['shared']* kernel_size)))
            else:
                raise NotImplementedError
            if trainable_scale_shared:
                self.lora_shared_scale = nn.Parameter(
                    torch.FloatTensor([lora_shared_scale]))
            else:
                self.lora_shared_scale = lora_shared_scale
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_shared_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_shared_B)
        if hasattr(self, "lora_tasks_A"):
            for task in self.tasks:
                nn.init.kaiming_uniform_(
                    self.lora_tasks_A[task], a=math.sqrt(5))
                nn.init.zeros_(self.lora_tasks_B[task])

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, x_tasks: Dict[str, torch.Tensor] = None):
        # TODO: handle merging
        pretrained = self.conv(x)

        #print('pretrained:{}'.format(pretrained.shape))
        if self.r == 0:
            return pretrained, None
        
        if self.shared_mode == 'matrix':
            lora = F.conv2d(x, (self.lora_shared_B @ self.lora_shared_A).view(self.conv.weight.shape) * self.lora_shared_scale, self.conv.bias,padding=self.padding,groups=self.groups)
            lora_tasks = {
                task: pretrained +(F.conv2d((x if x_tasks is None else x_tasks[task]),
                                (self.lora_tasks_B[task]@ self.lora_tasks_A[task]).view(self.conv.weight.shape)* self.lora_task_scale[task],
                                 self.conv.bias,padding=self.padding,groups=self.groups))for task in self.tasks} if self.tasks is not None else None

        elif self.shared_mode == 'matrixv2':
            lora = F.conv2d(x, (self.lora_shared_B @ self.lora_shared_A).view(self.conv.weight.shape) * self.lora_shared_scale, self.conv.bias,padding=self.padding,groups=self.groups)
            lora_tasks = {
                task: pretrained + lora + (F.conv2d((x if x_tasks is None else x_tasks[task]),
                                (self.lora_tasks_B[task]@ self.lora_tasks_A[task]).view(self.conv.weight.shape)* self.lora_task_scale[task],
                                 self.conv.bias,padding=self.padding,groups=self.groups))for task in self.tasks} if self.tasks is not None else None
        elif self.shared_mode == 'addition':
            lora_tasks = {
                task: pretrained + (F.conv2d((x if x_tasks is None else x_tasks[task]),(self.lora_tasks_B[task]@ 
                self.lora_tasks_A[task]).view(self.conv.weight.shape)* self.lora_task_scale[task],self.conv.bias,padding=self.padding,groups=self.groups))
                for task in self.tasks
            } if self.tasks is not None else None
            # lora_total = lora_tasks['cls'] + lora_tasks['reg']
            # lora = self.lora_norm(lora_total)

        if self.use_norm:
            pretrained = self.norm_layer(pretrained)
            for task in self.tasks:
                lora_tasks[task] = self.norm_layer(lora_tasks[task])
        return pretrained, lora_tasks


class CompatConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, x_tasks=None):
        return super().forward(x), None

class LoRACAModule(nn.Module):
    """Channel attention module"""

    def __init__(self, channels=64, reduction=1,  lora_config=None):
        super(LoRACAModule, self).__init__()
        self.tasks = lora_config.TASKS
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if lora_config.FC1_ENABLED:
            self.fc1 = MTLoRAConv(channels, channels // reduction, kernel_size=1,use_norm=False, use_act=False,
                                    r=lora_config.R_PER_TASK if lora_config.ENABLED else 0,lora_shared_scale = lora_config.SHARED_SCALE,
                                    lora_task_scale=lora_config.TASK_SCALE,lora_dropout=lora_config.DROPOUT,
                                    tasks=(lora_config.TASKS if (lora_config.ENABLED or lora_config.INTERMEDIATE_SPECIALIZATION) else None),
                                    trainable_scale_shared=lora_config.TRAINABLE_SCALE_SHARED,trainable_scale_per_task=lora_config.TRAINABLE_SCALE_PER_TASK,
                                    shared_mode=lora_config.SHARED_MODE)
        else:
            self.fc1 = CompatConv( channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        if lora_config.FC2_ENABLED:  
            self.fc2 = MTLoRAConv(channels // reduction, channels, kernel_size=1,use_norm=False, use_act=False,
                r=lora_config.R_PER_TASK if lora_config.ENABLED else 0,lora_shared_scale = lora_config.SHARED_SCALE,
                                    lora_task_scale=lora_config.TASK_SCALE,lora_dropout=lora_config.DROPOUT,
                                    tasks=(lora_config.TASKS if (lora_config.ENABLED or lora_config.INTERMEDIATE_SPECIALIZATION) else None),
                                    trainable_scale_shared=lora_config.TRAINABLE_SCALE_SHARED,trainable_scale_per_task=lora_config.TRAINABLE_SCALE_PER_TASK,
                                    shared_mode=lora_config.SHARED_MODE)
        else:
            self.fc2 = CompatConv(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_tasks=None):
        module_input = x
        x = self.avg_pool(x)
        x, x_tasks1 = self.fc1(x, x_tasks)
        if x_tasks1 is not None:
            for task in self.tasks:
                x_tasks1[task] = self.relu(x_tasks1[task])
        x = self.relu(x)
        x,  x_tasks2 = self.fc2(x,x_tasks1)
        if x_tasks2 is not None:
            for task in self.tasks:
                x_tasks2[task] = self.sigmoid(x_tasks2[task]) * module_input
        x = self.sigmoid(x)
        return module_input * x, x_tasks2


