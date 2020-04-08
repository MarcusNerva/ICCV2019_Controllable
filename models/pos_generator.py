import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../')
from models.encoders import Pos_encoder_two_fc
from models.decoder import Pos_decoder
from models.gate import to_contiguous
# from data.dataset import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import myopts

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_step(self, logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
        # logprobsf.shape == (beam_size, n_words) 表明上一个词输入lstm后生成的新词
        # beam_size beam search搜索时size大小
        # t 时间戳, 生成到了第几个词
        # beam_seq.shape == (beam_size, seq_length) 里面是一个个的数字, 用来表示beam_size个句子
        # beam_seq_logprobs.shape == (beam_size, seq_length) 里面是每个词生成的概率
        # state.shape == ((1, beam_size, rnn_size), (1, beam_size, rnn_size))
        # 为什么要state呢? 因为state是使lstm运转不可或缺的条件, 在生成好现在的新句子后, 我们需要每个beam所对应的state,
        # 将这个state和当前生成的新词输入到lstm中, 才能生成下一个新词

        ys, ix = torch.sort(logprobsf, dim=1, descending=True)
        # ys中每一行都是一个beam所能生成的下一个新词的概率分布,概率从大到小
        # ix中每一行都是一个beam所能生成的下一个新词,用数字表示,概率从大到小,具体概率值可以从ys相同位置找出
        candidates = []
        cols = min(beam_size, ys.size(1))
        rows = beam_size
        if t == 0:
            rows = 1
        for c in range(cols):
            for r in range(rows):
                local_logprob = ys[r, c]
                candidate_logprob = beam_logprobs_sum[r] + local_logprob
                candidates.append({'word': ix[r, c], 'beam': r, 'sum_probability':candidate_logprob, 'word_probability': local_logprob})
        candidates = sorted(candidates, key=lambda x: -x['sum_probability'])

        new_state = [_.clone() for _ in state]
        if t >= 1:
            beam_seq_prev = beam_seq[:, :t].clone()
            beam_seq_logprobs_prev = beam_seq_logprobs[:, :t].clone()
        for beam in range(beam_size):
            candidate = candidates[beam]
            if t >= 1:
                beam_seq[beam, :t] = beam_seq_prev[candidate['beam'], :t]
                beam_seq_logprobs[beam, :t] = beam_seq_logprobs_prev[candidate['beam'], :t]
            for state_ix in range(len(new_state)):
                new_state[state_ix][:, beam, :] = state[state_ix][:, candidate['beam'], :]
            beam_seq[beam, t] = candidate['word']
            beam_seq_logprobs[beam, t] = candidate['word_probability']
            beam_logprobs_sum[beam] = candidate['sum_probability']
        state = new_state
        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

    def beam_search(self, state, logprobs, feat, *args, **kwargs):
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 5)
        beam_seq = torch.LongTensor(beam_size, self.seq_length).zero_()
        beam_seq_logprobs = torch.FloatTensor(beam_size, self.seq_length).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []

        for t in range(self.seq_length):
            logprobs[:, 1] = logprobs[:, 1] - 1000.0
            beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates = self.beam_step(logprobsf=logprobs,
                                                                                               beam_size=beam_size,
                                                                                               t=t,
                                                                                               beam_seq=beam_seq,
                                                                                               beam_seq_logprobs=beam_seq_logprobs,
                                                                                               beam_logprobs_sum=beam_logprobs_sum,
                                                                                               state=state)

            for beam in range(beam_size):
                if beam_seq[beam, t] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[beam, :].clone(),
                        'log_probability': beam_seq_logprobs[beam, :].clone(),
                        'sum_probability': beam_logprobs_sum[beam]
                    }
                    done_beams.append(final_beam)
                    beam_logprobs_sum[beam] = -1000

            it = beam_seq[:, t]
            feat_ = feat.expand(it.size(0), feat.size(0), feat.size(1))
            logprobs, state = self.get_logprobs_state(it.to(self.device), feat_, state)

        done_beams = sorted(done_beams, key= lambda x: -x['sum_probability'])[:beam_size]
        return done_beams

class Pos_generator(CaptionModel):
    def __init__(self, opt):
        super(Pos_generator, self).__init__()
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

        self.category_size = opt.category_size
        self.word_embed_size = opt.word_embed_size
        self.rnn_size = opt.rnn_size
        self.visual_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_probability = opt.drop_probability
        self.seq_length = opt.seq_length
        self.scheduled_sample_probability = 0.0

        self.encoder = Pos_encoder_two_fc(opt=opt)
        self.decoder = Pos_decoder(opt=opt)
        self.video_embed_h0 = nn.Linear(self.visual_size, self.rnn_size)
        self.video_embed_c0 = nn.Linear(self.visual_size, self.rnn_size)
        self.video_embed_h1 = nn.Linear(self.visual_size, self.rnn_size)
        self.video_embed_c1 = nn.Linear(self.visual_size, self.rnn_size)

        self.embed = nn.Embedding(self.category_size, self.word_embed_size)
        self.logit = nn.Linear(self.rnn_size, self.category_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden_twolayers(self, feat, feat_mask):
        feat_ = torch.from_numpy(np.sum(feat.detach().cpu().numpy(), axis=1, dtype=np.float32))
        mask_ = torch.from_numpy(np.sum(feat_mask.detach().cpu().numpy(), axis=1, dtype=np.float32))
        feat_mean = (feat_ / mask_.unsqueeze(-1)).unsqueeze(0)
        feat_mean = feat_mean.to(self.device)
        state0 = (self.video_embed_h0(feat_mean), self.video_embed_c0(feat_mean))
        state1 = (self.video_embed_h1(feat_mean), self.video_embed_c1(feat_mean))
        return [state0, state1]

    def init_hidden(self, feat, feat_mask):
        feat_ = torch.from_numpy(np.sum(feat.detach().cpu().numpy(), axis=1, dtype=np.float32))
        mask_ = torch.from_numpy(np.sum(feat_mask.detach().cpu().numpy(), axis=1, dtype=np.float32))
        feat_mean = (feat_ / mask_.unsqueeze(-1)).unsqueeze(0)
        feat_mean = feat_mean.to(self.device)
        state0 = (self.video_embed_h0(feat_mean), self.video_embed_c0(feat_mean))
        return state0

    def forward(self, feats_rgb, feats_opfl, feat_mask, seq, seq_mask, cap_classes, new_mask):

        feats = self.encoder(feat0=feats_rgb, feat1=feats_opfl, feat_mask=feat_mask)

        seq = cap_classes
        seq_mask = new_mask
        # 此处是针对pos的generator, seq中具体是什么词我们并不关心

        batch_size = feats.size(0)
        state = self.init_hidden(feats, feat_mask)
        outputs_hidden = []
        outputs = []
        # 下面采用的是teach-force
        for i in range(cap_classes.size(1)):
            it = seq[:, i].clone()
            if i >= 1 and seq[:, i].detach().sum() == 0: break

            xt = self.embed(it)
            xt_mask = seq_mask[:, i].unsqueeze(1)
            output, state = self.decoder(xt, feats, xt_mask, state)
            output_category = torch.log_softmax(self.logit(output), dim=1)
            outputs_hidden.append(output)
            outputs.append(output_category)

        ret = torch.cat([_.unsqueeze(1) for _ in outputs], dim=1)
        ret = to_contiguous(ret)
        return ret

    def get_logprobs_state(self, it, feats, state):
        batch_size = it.size(0)
        xt = self.embed(it)
        xt_mask = torch.ones([batch_size, 1]).float()
        xt_mask = xt_mask.to(self.device)

        output, state = self.decoder(xt, xt_mask, feats, state)
        logprobs = torch.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def beam_sample(self, feats, feat_masks, opt={}):
        beam_size = opt.get('beam_size', 5)
        batch_size = feats.size(0)

        seq = torch.LongTensor(batch_size, self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, self.seq_length)

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            feat = feats[k]
            feat = feat.expand(beam_size, feat.size(0), feat.size(1))
            feat_mask = feat_masks[k]
            feat_mask = feat_mask.expand(beam_size, feat_mask.size(0))
            state = self.init_hidden(feat, feat_mask)

            it = feats.detach().new(beam_size).long().zero_()
            xt = self.embed(it).to(self.device)
            xt_mask = torch.ones([beam_size, 1]).float().to(self.device)
            output, state = self.decoder(xt, xt_mask, feat, state)
            logprobs = torch.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = self.beam_search(state, logprobs, feats[k], opt=opt)
            seq[k, :] = self.done_beams[k][0]['seq']
            seqLogprobs[k, :] = self.done_beams[k][0]['log_probability']
            # 此处为什么有[0]呢? 因为返回的是top beam_size个选择, 我们选择最优的那个, 也即[0]

        return seq, seqLogprobs

    def sample(self, feats_rgb, feats_opfl, feat_mask, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        feats = self.encoder(feats_rgb, feats_opfl, feat_mask)

        if beam_size > 1:
            return self.beam_sample(feats, feat_mask, opt)
        batch_size = feats.size(0)
        state = self.init_hidden(feats, feat_mask)
        seq = []
        seqLogprobs = []
        collect_states = []
        collect_masks = []
        for t in range(self.seq_length + 1):
            if t == 0:
                it = feats.data.new_zeros(batch_size).long()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                prob_prev = torch.exp(torch.div(logprobs.detach(), temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).to(self.device)
                sampleLogprobs = logprobs.gather(1, it)

            xt = self.embed(it)

            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0: break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

            if t == 0:
                xt_mask = torch.ones([batch_size, 1]).float()
            else:
                xt_mask = unfinished.unsqueeze(-1).float()
            xt_mask = xt_mask.to(self.device)
            output, state = self.decoder(xt, feats, xt_mask, state)
            logprobs = torch.log_softmax(self.logit(output), dim=1)
            collect_states.append(state[0][-1])
            collect_masks.append(xt_mask)
        collect_states = torch.stack(collect_states, dim=1)
        collect_masks = torch.cat(collect_masks, dim=1)
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), collect_states, collect_masks


# class Opt_stub(object):
#     def __init__(self):
#         super(Opt_stub, self).__init__()
#         self.seed = 1
#         self.num_layers = 1
#         self.feat0_size = 1536
#         self.feat1_size = 1024
#         self.rnn_size = 512
#         self.pos_size = 512
#         self.drop_probability = 0.5
#         self.activity_fn = 'ReLU'
#         self.seqlength = 28
#         self.feat_K = 28
#         self.batch_size = 64
#         self.vocab_size = 20000
#         self.word_embed_size = 468
#         self.att_size = 1536
#         self.category_size = 14
#         self.data_path = '/Users/bismarck/PycharmProjects/ICCV2019_Controllable/data/'

# if __name__ == '__main__':
#     opt = myopts.parse_opt()
#     model = pos_generator(opt)
#     train_dataset, valid_dataset, test_dataset = load_dataset_pos(opt=opt)
#     valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_pos)
#     model.to(device)
#     model.train()
#
#     for i, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, lens, gts, video_id) in enumerate(valid_dataloader):
#         seq, seqlogprob, c0, c1 = model.max_sample(feats0, feats1, feat_mask, {'sample_max': True})
#         if i % 4 == 0:
#             print(seq.shape)
#
#     pass