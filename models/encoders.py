import torch
import numpy as np
import torch.nn as nn
import sys
sys.path.append('../')
from data import *
from models.gate import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder_two_fc(nn.Module):
    def __init__(self, opt):
        super(Encoder_two_fc, self).__init__()
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

        self.feat_rgb_size = opt.feat0_size
        self.feat_opfl_size = opt.feat1_size
        self.rnn_size = opt.rnn_size
        self.drop_probability = opt.drop_probability

        self.visual_emb_rgb = nn.Sequential(nn.Linear(self.feat_rgb_size, self.rnn_size),
                                            nn.BatchNorm1d(self.rnn_size),
                                            nn.ReLU(True))
        self.visual_emb_opfl = nn.Sequential(nn.Linear(self.feat_opfl_size, self.rnn_size),
                                             nn.BatchNorm1d(self.rnn_size),
                                             nn.ReLU(True))
        self.dropout = nn.Dropout(self.drop_probability)
        self.lstmcell_rgb = nn.LSTMCell(self.rnn_size, self.rnn_size)
        self.lstmcell_opfl = nn.LSTMCell(self.rnn_size, self.rnn_size)
        self.gate_rgb = Gate(seed=opt.seed, source_size=self.rnn_size, target_size=self.rnn_size, drop_lm=self.drop_probability)
        self.gate_opfl = Gate(seed=opt.seed, source_size=self.rnn_size, target_size=self.rnn_size, drop_lm=self.drop_probability)
        self.fuse = Fusion(seed=opt.seed, feat1_size=self.rnn_size, feat2_size=self.rnn_size, output_size=self.rnn_size, drop_lm=self.drop_probability, activity_fn=opt.activity_fn)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_hidden(self, batch_size):
        h_size = (batch_size, self.rnn_size)
        h_0 = torch.FloatTensor(*h_size).zero_()
        c_0 = torch.FloatTensor(*h_size).zero_()
        # print(self.device)
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        print('h_0.device is ', h_0.device)
        print('c_0.device is ', c_0.device)
        return (h_0, c_0)

    def forward(self, feat0, feat1, feat_mask):#Can I remove feat_mask ?
        batch_size, length = feat1.size(0), feat1.size(1)
        embed_feat0 = self.visual_emb_rgb(feat0.view(-1, feat0.size(-1)))
        embed_feat1 = self.visual_emb_opfl(feat1.view(-1, feat1.size(-1)))

        embed_feat0 = to_contiguous(embed_feat0)
        embed_feat0 = embed_feat0.view(batch_size, length, -1)
        embed_feat0 = self.dropout(embed_feat0)
        feat0_init_state = self.init_hidden(batch_size)
        # feat0_init_state = feat0_init_state.to(device)

        embed_feat1 = to_contiguous(embed_feat1)
        embed_feat1 = embed_feat1.view(batch_size, length, -1)
        embed_feat1 = self.dropout(embed_feat1)
        feat1_init_state = self.init_hidden(batch_size)
        # feat1_init_state = feat1_init_state.to(device)

        out_feats0, out_feats1 = [], []
        # h0, c0 = feat0_init_state[0].to(self.device), feat0_init_state[1].to(self.device)
        # h1, c1 = feat1_init_state[0].to(self.device), feat1_init_state[1].to(self.device)

        h0, c0 = feat0_init_state[0], feat0_init_state[1]
        h1, c1 = feat1_init_state[0], feat1_init_state[1]
        print('h0.device is ', h0.device)
        print('c0.device is ', c0.device)
        print('h1.device is ', h1.device)
        print('c1.device is ', c1.device)

        for i in range(length):
            input_0 = embed_feat0[:, i, :]
            input_1 = embed_feat1[:, i, :]
            mask = feat_mask[:, i]

            h0, c0 = self.lstmcell_rgb(input_0, (h0, c0))
            h1, c1 = self.lstmcell_opfl(input_1, (h1, c1))
            h0 = h0 * mask.unsqueeze(-1)
            c0 = c0 * mask.unsqueeze(-1)
            h1 = h1 * mask.unsqueeze(-1)
            c1 = c1 * mask.unsqueeze(-1)

            gate0 = self.gate_rgb(h1, h0)
            gate1 = self.gate_opfl(h0, h1)
            out_feats0.append(gate0)
            out_feats1.append(gate1)
        out_feats0 = torch.cat([item.unsqueeze(1) for item in out_feats0], dim=1)
        out_feats1 = torch.cat([item.unsqueeze(1) for item in out_feats1], dim=1)
        ret = self.fuse(out_feats0, out_feats1)
        return ret

# class Opt_stub(object):
#     def __init__(self):
#         super(Opt_stub, self).__init__()
#         self.seed = 1
#         self.feat0_size = 1536
#         self.feat1_size = 1024
#         self.rnn_size = 512
#         self.drop_probability = 0.5
#         self.activity_fn = 'ReLU'
#         self.seqlength = 28
#         self.feat_K = 28
#         self.batch_size = 64
#         self.vocab_size = 20000
#         self.data_path = '/Users/bismarck/PycharmProjects/ICCV2019_Controllable/data/'


# if __name__ == '__main__':
#     opt = Opt_stub()
#     encoder = Encoder_two_fc(opt=opt)
#     pos_train_dataset, pos_valid_dataset, pos_test_dataset = load_dataset_pos(opt=opt)
#     pos_validloader = DataLoader(pos_valid_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_pos)
#     for i, (data, caps, caps_mask, cap_classes, class_masks, feats1, feats2, feat_mask, lens, gts, image_id) in enumerate(pos_validloader):
#         if i % 30 == 0:
#             ans = encoder(feats1, feats2, feat_mask)
#             print('ret.shape == ', ans.shape)
#     pass