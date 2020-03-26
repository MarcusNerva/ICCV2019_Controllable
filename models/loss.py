import torch
import torch.nn as nn
import numpy as np
from models.pos_generator import *

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input.shape == (batch_size, seq_len + 1, n_words)
        # target.shape == (batch_size, seq_len + 1)
        # mask.shape == (batch_size, seq_len + 1)

        input = to_contiguous(input).view(-1, input.size(-1))
        target = torch.cat([target[:, 1:], target[:, 0].unsqueeze(1)], dim=1)
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        output = -1. * input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

class ClassifierCriterion(nn.Module):
    def __init__(self):
        super(ClassifierCriterion, self).__init__()

    def forward(self, input, target, mask, class_mask=None):
        # input.shape == (batch_size, seq_len + 1, n_classes)
        # target.shape == (batch_size, seq_len + 1)
        # mask.shape == (batch_size, seq_len + 1)

        # print('input.shape == ', input.shape)
        # print('target.shape == ', target.shape)
        # print('mask.shape == ', mask.shape)
        input = to_contiguous(input).view(-1, input.size(-1))
        target = torch.cat([target[:, 1:], target[:, 0].unsqueeze(1)], dim=1)
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        # print('input.shape == ', input.shape)
        # print('target.shape == ', target.shape)
        # print('mask.shape == ', mask.shape)
        output = -1. * input.gather(1, target) * mask
        # 此处其实还是交叉熵
        if class_mask is None:
            output = torch.sum(output) / torch.sum(mask)
        else:
            class_mask = to_contiguous(class_mask).view(-1, 1)
            print('output.device is ', output.device)
            print('class_mask.device is ', class_mask.device)
            output = output * class_mask
            output = torch.sum(output) / torch.sum(mask)
        return output

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], dim=1)).view(-1)
        output = -input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

