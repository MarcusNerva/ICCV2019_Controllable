import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from data.dataset import *
import myopts

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class Gate(nn.Module):
    def __init__(self, seed, source_size, target_size, drop_lm=0.5, simple=True):
        super(Gate, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # 分别为cpu和gpu设定随机种子
        self.source_size = source_size
        self.middle_size = 2 * source_size
        self.target_size = target_size
        self.drop_prob_lm = drop_lm

        if simple:
            self.gate = nn.Sequential(nn.Linear(self.source_size, self.target_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        else:
            self.gate = nn.Sequential(nn.Linear(self.source_size, self.middle_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm),
                                      nn.Linear(self.middle_size, self.target_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))

    def forward(self, source, target):
        gate = self.gate(source)
        ret = gate * target + target
        return ret

class Fusion(nn.Module):
    def __init__(self, seed, feat1_size, feat2_size, output_size, drop_lm=0.5, activity_fn=None):
        super(Fusion, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.feat1_size = feat1_size
        self.feat2_size = feat2_size
        self.output_size = output_size
        self.drop_lm = drop_lm
        self.activity_fn = getattr(nn, activity_fn)()

        self.fuse = nn.Sequential(nn.Linear(self.feat1_size + self.feat2_size, self.output_size),
                                  self.activity_fn,
                                  nn.Dropout(self.drop_lm))

    def forward(self, feat1, feat2):
        input = torch.cat([feat1, feat2], dim=-1)
        input = to_contiguous(input)
        ret = self.fuse(input)
        return ret


# if __name__ == '__main__':
    # source_size = 1536
    # target_size = 2000
    # drop_lm = 0.5
    # source = torch.randn(10, 28, 1536)
    # target = torch.randn(10, 28, 2000)
    # seed = 1
    # gate = Gate(seed=seed, source_size=source_size, target_size=target_size, drop_lm=drop_lm, simple=True)
    # ans = gate(source=source, target=target)
    # print('ans.shape == ', ans.shape)

    # seed = 1
    # feat1_size = 1536
    # feat2_size = 2000
    # output_size = 2001
    # drop_lm = 0.5
    # activity_fn = 'ReLU'
    # fuse = Fusion(seed=seed, feat1_size=feat1_size, feat2_size=feat2_size, output_size=output_size, drop_lm=drop_lm, activity_fn=activity_fn)
    # feat1 = torch.randn(10, 28, feat1_size)
    # feat2 = torch.randn(10, 28, feat2_size)
    # ans = fuse(feat1, feat2)
    # print('after fuse, shape is ', ans.shape)
    #
    # pass
