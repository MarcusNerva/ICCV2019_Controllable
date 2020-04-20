import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import sys
sys.path.append('../')
import os
from data import *
from models.gate import *
from models.encoders import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class One_input_lstm(nn.Module):
    def __init__(self, input_size, rnn_size, drop_probability=0.5):
        super(One_input_lstm, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.drop_probability = drop_probability
        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        if self.drop_probability is not None:
            self.dropout = nn.Dropout(self.drop_probability)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.i2h.weight.data)
        nn.init.xavier_uniform_(self.h2h.weight.data)

    def forward(self, input, state, mask=None):
        all_input_sums = self.i2h(input) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(dim=1, start=0, length=3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(dim=1, start=0, length=self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(dim=1, start=self.rnn_size, length=self.rnn_size)
        out_gate = sigmoid_chunk.narrow(dim=1, start=self.rnn_size * 2, length=self.rnn_size)
        tanh_chunk = all_input_sums.narrow(dim=1, start=self.rnn_size * 3, length=self.rnn_size)
        in_transform = torch.tanh(tanh_chunk)
        state_c = forget_gate * state[1][-1] + in_gate * in_transform
        if mask is not None:
            state_c = state_c * mask + state[1][-1] * (1 - mask)
        state_h = out_gate * torch.tanh(state_c)
        if mask is not None:
            state_h = state_h * mask + state[0][-1] * (1 - mask)
        if self.drop_probability is not None:
            state_h = self.dropout(state_h)

        output = state_h
        return output, (state_h.unsqueeze(0), state_c.unsqueeze(0))

class Two_inputs_lstmcell(nn.Module):
    def __init__(self, input_size, visual_size, rnn_size, drop_probabilily=0.5):
        super(Two_inputs_lstmcell, self).__init__()
        self.input_size = input_size
        self.visual_size = visual_size
        self.rnn_size = rnn_size
        self.drop_probability = drop_probabilily
        self.i2h = nn.Linear(self.input_size, self.rnn_size * 4)
        self.v2h = nn.Linear(self.visual_size, self.rnn_size * 4)
        self.h2h = nn.Linear(self.rnn_size, self.rnn_size * 4)
        if self.drop_probability is not None:
            self.dropout = nn.Dropout(drop_probabilily)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.i2h.weight.data)
        nn.init.xavier_uniform_(self.v2h.weight.data)
        nn.init.xavier_uniform_(self.h2h.weight.data)

    def forward(self, input0, input1, state, mask=None):
        all_input_sums = self.i2h(input0) + self.v2h(input1) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(dim=1, start=0, length=self.rnn_size * 3)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(dim=1, start=0, length=self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(dim=1, start=self.rnn_size, length=self.rnn_size)
        out_gate = sigmoid_chunk.narrow(dim=1, start=self.rnn_size * 2, length=self.rnn_size)
        tanh_chunk = all_input_sums.narrow(dim=1, start=self.rnn_size * 3, length=self.rnn_size)
        in_transform = torch.tanh(tanh_chunk)

        state_c = forget_gate * state[1][-1] + in_gate * in_transform
        if mask is not None:
            state_c = state_c * mask + state[1][-1] * (1 - mask)
        state_h = out_gate * torch.tanh(state_c)
        if mask is not None:
            state_h = state_h * mask + state[0][-1] * (1 - mask)

        if self.drop_probability is not None:
            state_h = self.dropout(state_h)
        output = state_h

        return output, (state_h.unsqueeze(0), state_c.unsqueeze(0))

class Pos_decoder(nn.Module):
    def __init__(self, opt):
        super(Pos_decoder, self).__init__()
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        self.word_embed_size = opt.word_embed_size
        self.visual_size = opt.rnn_size
        self.rnn_size = opt.rnn_size
        self.att_size = opt.att_size
        self.drop_probability = opt.drop_probability
        self.lstmcell = Two_inputs_lstmcell(self.word_embed_size, self.visual_size, self.rnn_size, self.drop_probability)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.v2a = nn.Linear(self.visual_size, self.att_size)
        self.to_e = nn.Linear(self.att_size, 1)

        # self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.h2a.weight.data)
        nn.init.xavier_uniform_(self.v2a.weight.data)
        nn.init.xavier_uniform_(self.to_e.weight.data)

    def forward(self, word, visual_feat, word_mask, state):
        # print('-----------visual_feat.shape is ', visual_feat.shape, '-----------')
        # print('++++++++++ part0.shape is ', self.h2a(state[0][-1]).unsqueeze(1).shape, ' ++++++++++++')
        # print('++++++++++ part1.shape is ', self.v2a(visual_feat).shape, ' ++++++++++++')
        alpha = self.h2a(state[0][-1]).unsqueeze(1) + self.v2a(visual_feat)
        # print('alpha.shape == ', alpha.shape)
        e = self.to_e(torch.tanh(alpha))
        e = e.transpose(1, 0)
        e = torch.softmax(e, dim=0).transpose(1, 0)
        atten_visual_feat = e * visual_feat
        atten_visual_feat = torch.sum(atten_visual_feat, dim=1)

        output, state = self.lstmcell(word, atten_visual_feat, state, word_mask)
        return output, state

class Describe_decoder(nn.Module):
    def __init__(self, opt):
        super(Describe_decoder, self).__init__()
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        self.word_embed_size = opt.word_embed_size
        self.visual_size = opt.rnn_size
        self.pos_size = opt.pos_size
        self.rnn_size = opt.rnn_size
        self.att_size = opt.att_size
        self.drop_probability = opt.drop_probability
        self.gate = Gate(opt.seed, self.word_embed_size, self.pos_size)
        self.lstm0 = Two_inputs_lstmcell(input_size=self.word_embed_size, visual_size=self.pos_size, rnn_size=self.rnn_size, drop_probabilily=opt.drop_probability)
        self.lstm1 = Two_inputs_lstmcell(input_size=self.rnn_size, visual_size=self.visual_size, rnn_size=self.rnn_size, drop_probabilily=opt.drop_probability)
        self.dropout = nn.Dropout(opt.drop_probability)
        self.h2a = nn.Linear(2 * self.rnn_size, self.att_size)
        self.v2a = nn.Linear(self.visual_size, self.att_size)
        self.to_e = nn.Linear(self.att_size, 1)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.h2a.weight.data)
        nn.init.xavier_uniform_(self.v2a.weight.data)
        nn.init.xavier_uniform_(self.to_e.weight.data)

    def forward(self, word, word_mask, visual_info, pos_feat, state):
        assert len(state) == 2, 'input parameters state expect a list with 2 elements'
        state_0, state_1 = state[0], state[1]

        part0 = self.h2a(torch.cat([state_0[0][-1], state_1[0][-1]], dim=1)).unsqueeze(1)
        part1 = self.v2a(visual_info)
        temp = torch.tanh(part0 + part1)
        e = self.to_e(temp)
        alpha = torch.softmax(e.transpose(2, 1), dim=-1).transpose(2, 1)
        attention_visual_info = alpha * visual_info
        attention_visual_info = torch.sum(attention_visual_info, dim=1)
        psi = pos_feat
        psi_bar = self.gate(word, psi)
        output0, state_0 = self.lstm0(input0=word, input1=psi_bar, state=state_0, mask=word_mask)
        output1, state_1 = self.lstm1(input0=output0, input1=attention_visual_info, state=state_1, mask=word_mask)

        two_layer_state = [state_0, state_1]
        return output1, two_layer_state

class Opt_stub(object):
    def __init__(self):
        super(Opt_stub, self).__init__()
        self.seed = 1
        self.feat0_size = 1536
        self.feat1_size = 1024
        self.rnn_size = 512
        self.pos_size = 512
        self.drop_probability = 0.5
        self.activity_fn = 'ReLU'
        self.seqlength = 28
        self.feat_K = 28
        self.batch_size = 64
        self.vocab_size = 20000
        self.word_embed_size = 468
        self.att_size = 1536
        self.data_path = '/Users/bismarck/PycharmProjects/ICCV2019_Controllable/data/'

# if __name__ == '__main__':
#     opt = Opt_stub()
#
#     word = torch.randn((opt.batch_size, opt.word_embed_size))
#     word_mask = torch.ones((opt.batch_size, 1), dtype=torch.float)
#     visual_feat = torch.randn((opt.batch_size, opt.seqlength, opt.rnn_size))
#     pos_feat = torch.randn((opt.batch_size, opt.pos_size))
#     state = [(torch.randn([1, opt.batch_size, opt.rnn_size]), torch.randn([1, opt.batch_size, opt.rnn_size])),
#              (torch.randn([1, opt.batch_size, opt.rnn_size]), torch.randn([1, opt.batch_size, opt.rnn_size]))]
#     model = Describe_decoder(opt=opt)
#     output, two_layer_state = model(word=word, word_mask=word_mask, visual_info=visual_feat, pos_feat=pos_feat, state=state)
#     print('output.shape == ', output.shape)
#     print('state.shape == ', two_layer_state[0][0].shape)

#
#     input_size = 1536
#     rnn_size = 512
#     input0 = torch.randn((opt.batch_size, input_size))
#     input1 = torch.randn((opt.batch_size, rnn_size))
#     state_h = torch.randn((1, opt.batch_size, rnn_size))
#     state_c = torch.randn((1, opt.batch_size, rnn_size))
#     model = One_input_lstm(input_size=input_size, rnn_size=rnn_size)
#     model = Two_inputs_lstmcell(input_size=input_size, visual_size=rnn_size, rnn_size=rnn_size)
#     output, (state_h, state_c) = model(input0=input0, input1=input1, state=(state_h, state_c))
#     print('output.shape == ', output.shape, ' state_h.shape == ', state_h.shape, ' state_c == ', state_c.shape)


    # decoder = Pos_decoder(opt=opt)
    # state_h, state_c = torch.randn((1, opt.batch_size, opt.rnn_size)), torch.randn((1, opt.batch_size, opt.rnn_size))
    # word = torch.randn((opt.batch_size, opt.word_embed_size))
    # visual_feat = torch.randn((opt.batch_size, opt.seqlength, opt.rnn_size))
    # output, (state_h, state_c) = decoder(word, visual_feat, None, (state_h, state_c))
    # print('output.shape == ', output.shape)
    # print('state_h.shape == ', state_h.shape)
    # print('state_c.shape == ', state_c.shape)

    # pass