import torch
import torch.nn as nn
import torch.nn.init as init
import sys
sys.path.append('../')
from models.decoder import *
from models.encoders import *
from models.gate import *
from torch.utils.data import DataLoader
import myopts

class Caption_generator(nn.Module):
    def __init__(self, opt):
        super(Caption_generator, self).__init__()
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

        self.vocab_size = opt.vocab_size
        self.category_size = opt.category_size
        self.word_embed_size = opt.word_embed_size
        self.rnn_size = opt.rnn_size
        self.visual_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_probability = opt.drop_probability
        self.seq_length = opt.seq_length
        self.sample_probability = 0.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = Encoder_two_fc(opt=opt)
        self.decoder = Describe_decoder(opt=opt)
        self.embed = nn.Embedding(self.vocab_size, self.word_embed_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.rnn_size, 128),
            nn.ReLU(),
            nn.Dropout(self.drop_probability),
            nn.Linear(128, self.category_size)
        )
        self.state_init_h0 = nn.Linear(self.visual_size, self.rnn_size)
        self.state_init_c0 = nn.Linear(self.visual_size, self.rnn_size)
        self.state_init_h1 = nn.Linear(self.visual_size, self.rnn_size)
        self.state_init_c1 = nn.Linear(self.visual_size, self.rnn_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)

    def init_hidden(self, feat, feat_mask):
        feat_ = torch.from_numpy(np.sum(feat.detach().cpu().numpy(), axis=1, dtype=np.float32))
        mask_ = torch.from_numpy(np.sum(feat_mask.detach().cpu().numpy(), axis=1, dtype=np.float32))
        feat_mean = (feat_ / mask_.unsqueeze(-1)).unsqueeze(0)
        feat_mean = feat_mean.to(self.device)
        state0 = (self.state_init_h0(feat_mean), self.state_init_c0(feat_mean))
        state1 = (self.state_init_h1(feat_mean), self.state_init_c1(feat_mean))
        return [state0, state1]

    def forward(self, feats_rgb, feats_opfl, feat_mask, pos_feats, seq, seq_mask):
        feats = self.encoder(feat0=feats_rgb, feat1=feats_opfl, feat_mask=feat_mask)
        batch_size = feats.size(0)
        output_hidden = []
        outputs, categories = [], []
        state = self.init_hidden(feats, feat_mask)
        for i in range(seq.size(1)):
            if i >= 1 and seq[:, i].detach().sum() == 0:
                break

            if self.training and i >= 1 and self.sample_probability > 0.0:
                sample_prob = feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.sample_probability
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].clone()
                    prob_prev = torch.exp(outputs[-1].detach())
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()

            xt = self.embed(it)
            xt_mask = seq_mask[:, i].unsqueeze(1)
            output, state = self.decoder(xt, xt_mask, feats, pos_feats, state)
            output_hidden.append(output)
            output_word = torch.log_softmax(self.logit(output), dim=1)
            output_category = torch.log_softmax(self.classifier(output), dim=1)
            outputs.append(output_word)
            categories.append(output_category)

        ret_seq_word = torch.cat([_.unsqueeze(1) for _ in outputs], dim=1).contiguous()
        ret_seq_category =torch.cat([_.unsqueeze(1) for _ in categories], dim=1).contiguous()
        return ret_seq_word, ret_seq_category

    def beam_step(self, logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
        # logprobs.shape == (beam_size, n_words)
        # beam_seq.shape == (beam_size, seq_length)
        # beam_seq_logprobs.shape == (beam_size, seq_length)
        # beam_logprobs_sum.shape == (beam_size)
        probs, idx = torch.sort(logprobs, dim=1, descending=True)
        candidates = []
        rows = beam_size if t >= 1 else 1
        # 在刚开始时, 每行都一样, 所以只考虑1行了
        cols = min(beam_size, probs.size(1))
        # 这里为什么要令cols = min(beam_size, probs.size(1)) 按道理不应该是要把所有的组合都试一下吗?
        # 此处运用了剪枝, 因为我们每次都取这beam_size * n_words这么多组合的前beam_size个, 对于一个beam来说,
        # 就算这一步最终答案中前beam_size个都产自这个beam, 我们也只需要考虑它的前beam_size个, 不用考虑所有的n_words
        for r in range(rows):
            for c in range(cols):
                tmp_logprob = probs[r, c]
                tmp_sum = beam_logprobs_sum[r] + tmp_logprob
                tmp_idx = idx[r, c]
                candidates.append({'sum': tmp_sum, 'logprob': tmp_logprob, 'ix': tmp_idx, 'beam': r})

        candidates = sorted(candidates, key=lambda x: -x['sum'])
        prev_seq = beam_seq[:, :t].clone()
        prev_seq_probs = beam_seq_logprobs[:, :t].clone()
        prev_logprobs_sum = beam_logprobs_sum.clone()
        new_state0 = [_.clone() for _ in state[0]]
        new_state1 = [_.clone() for _ in state[1]]

        for i in range(beam_size):
            candidate_i = candidates[i]
            beam = candidate_i['beam']
            ix = candidate_i['ix']
            logprob = candidate_i['logprob']
            # print('beam_seq[i].shape is ', beam_seq[i].shape)
            # print('prev_seq[beam, :].shape is ', prev_seq[beam, :].shape)
            # if t > 0:
            beam_seq[i, :t] = prev_seq[beam, :]
            beam_seq_logprobs[i, :t] = prev_seq_probs[beam, :]
            beam_seq[i, t] = ix
            beam_seq_logprobs[i, t] = logprob
            beam_logprobs_sum[i] = prev_logprobs_sum[beam] + logprob
            for j in range(len(new_state0)):
                new_state0[j][:, i, :] = state[0][j][:, beam, :]
                new_state1[j][:, i, :] = state[1][j][:, beam, :]

        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, [new_state0, new_state1]

    def beam_search(self, state, feat, pos_feat, *args, **kwargs):
        # logprobs.shape == (beam_size, n_words)
        # state == (state0, state1)
        # state0 == (h0, c0)
        # h0.shape == c0.shape
        # h0.shape == (1, batch_size, rnn_size)
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 5)

        beam_seq = torch.LongTensor(beam_size, self.seq_length).zero_()
        beam_seq_logprobs = torch.FloatTensor(beam_size, self.seq_length).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)
        ret = []

        it = torch.LongTensor(beam_size).zero_().to(self.device)
        xt = self.embed(it).to(self.device)
        xt_mask = torch.ones([beam_size, 1]).float().to(self.device)
        output, state = self.decoder(xt, xt_mask, feat, pos_feat, state)
        logprob = torch.log_softmax(self.logit(output), dim=1)

        for t in range(self.seq_length):
            logprob[:, 1] = logprob[:, 1] - 1000.0
            beam_seq, beam_seq_logprobs, beam_logprobs_sum, state = self.beam_step(logprobs=logprob,
                                                                                   beam_size=beam_size,
                                                                                   t=t,
                                                                                   beam_seq=beam_seq,
                                                                                   beam_seq_logprobs=beam_seq_logprobs,
                                                                                   beam_logprobs_sum=beam_logprobs_sum,
                                                                                   state=state)

            for j in range(beam_size):
                if beam_seq[j, t] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[j, :].clone(),
                        'seq_logprob': beam_seq_logprobs[j, :].clone(),
                        'sum_logprob': beam_logprobs_sum[j].clone()
                    }
                    ret.append(final_beam)
                    beam_logprobs_sum[j] = -1000.0

            it = beam_seq[:, t].to(self.device)
            xt = self.embed(it).to(self.device)
            xt_mask = torch.ones([beam_size, 1]).float().to(self.device)
            output, state = self.decoder(xt, xt_mask, feat, pos_feat, state)
            logprob = torch.log_softmax(self.logit(output), dim=1)

        ret = sorted(ret, key=lambda x: -x['sum_logprob'])[:beam_size]
        return ret

    def sample_beam(self, feats, feat_masks, pos_feats, opt={}):
        beam_size = opt.get('beam_size', 5)
        batch_size = feats.size(0)

        seq = torch.LongTensor(batch_size, self.seq_length).zero_()
        seq_probabilities = torch.FloatTensor(batch_size, self.seq_length)
        done_beam = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            feat = feats[i]
            feat = feat.expand(beam_size, feat.size(0), feat.size(1))
            feat_mask = feat_masks[i]
            feat_mask = feat_mask.expand(beam_size, feat_mask.size(0))
            pos_feat = pos_feats[i]
            # print('pos_feat.shape is ', pos_feat.shape)
            pos_feat = pos_feat.expand(beam_size, pos_feat.size(0))
            state = self.init_hidden(feat, feat_mask)

            done_beam[i] = self.beam_search(state, feat, pos_feat, opt=opt)
            seq[i] = done_beam[i][0]['seq']
            seq_probabilities[i] = done_beam[i][0]['seq_logprob']

        return seq, seq_probabilities

    def sample(self, feats_rgb, feats_opfl, feat_mask, pos_feat, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        feats = self.encoder(feats_rgb, feats_opfl, feat_mask)
        if beam_size > 1:
            return self.sample_beam(feats, feat_mask, pos_feat, opt=opt)

        batch_size = feats.size(0)
        state = self.init_hidden(feats, feat_mask)
        seq = []
        seq_probabilities = []

        for t in range(self.seq_length + 1):
            if t == 0:
                it = feats.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(log_probabilities, 1)
                it = it.view(-1).long()
            else:
                prev_probabilities = torch.exp(torch.div(log_probabilities, temperature))
                it = torch.multinomial(prev_probabilities, 1)
                sampleLogprobs = log_probabilities.gather(1, it)
                it = it.view(-1).long()

            xt = self.embed(it)

            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                    # 这里unfinished 是积累值, 已完成部分越来越大
                if unfinished.sum() == 0: break
                it = it * unfinished.type_as(it)
                # 这里必须再乘一次, 确保有效部分和unfinished保持一致
                seq.append(it)
                seq_probabilities.append(sampleLogprobs.view(-1))

            if t == 0:
                xt_mask = torch.ones([batch_size, 1]).float()
            else:
                xt_mask = unfinished.unsqueeze(-1).float()

            xt = xt.to(self.device)
            xt_mask = xt_mask.to(self.device)
            output, state = self.decoder(xt, xt_mask, feats, pos_feat, state)
            log_probabilities = torch.log_softmax(self.logit(output), dim=1)

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seq_probabilities], 1)

# if __name__ == '__main__':
#     opt = myopts.parse_opt()
#     model = Caption_generator(opt)
    # size0 = [opt.batch_size, 28, 1536]
    # size1 = [opt.batch_size, 28, 1024]
    # size2 = [opt.batch_size, opt.rnn_size]
    # feats_rgb = torch.randn(size0)
    # feats_opfl = torch.randn(size1)
    # pos_feats = torch.randn(size2)
    # feat_mask = torch.ones([opt.batch_size, opt.seq_length])
    # seq = torch.ones([opt.batch_size, opt.seq_length])
    # seq_mask = torch.ones([opt.batch_size, opt.seq_length])
    # ret_seq_word, ret_seq_category = model(feats_rgb, feats_opfl, feat_mask, pos_feats, seq, seq_mask)
    # train_dataset, test_dataset, valid_dataset = load_dataset_cap(opt)
    # valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
    # for i, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, pos_feats, lens, gts, video_id) in enumerate(valid_loader):
        # print('feats0.shape is ', feats0.shape)
        # print('feats1.shape is ', feats1.shape)
        # print('feat_mask.shape is ', feat_mask.shape)
        # print('pos_feats.shape is ', pos_feats.shape)
        # print('caps.shape is ', caps.shape)
        # print('caps_mask is ', caps_mask.shape)
        # if i % 20 == 0:
        #     print('########now runs .forward()########')
        #     seq_word, seq_category = model(feats0, feats1, feat_mask, pos_feats, caps, caps_mask)
        #     print('########now runs .sample_beam()########')
        #     seq, seq_probabilities = model.sample(feats0, feats1, feat_mask, pos_feats, {'beam_size': 5})
        #     print('#########now runs .sample()########')
        #     seq, seq_probabilities = model.sample(feats0, feats1, feat_mask, pos_feats, {})
        # print('seq_word.shape is ', seq_word.shape)
        # print('seq_category.shape is ', seq_category.shape)
    # pass