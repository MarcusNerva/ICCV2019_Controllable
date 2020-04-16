import sys
sys.path.append('../')
import os
import torch
from torch.utils.data import DataLoader
from data.dataset import load_dataset_cap, collate_fn_cap, load_pkl, get_caps
from infersent_model import InferSent
from eval.eval_cap import cosine
import myopts
import math

EPS = 1e-4

if __name__ == '__main__':
    opt = myopts.parse_opt()
    model_version = 1
    MODEL_PATH = os.path.join(opt.infersent_model_path, 'infersent%s.pkl' % model_version)
    params_model = {
        'bsize': 64,
        'word_emb_dim': 300,
        'enc_lstm_dim': 2048,
        'pool_type': 'max',
        'dpout_model': 0.0,
        'version': model_version
    }
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    W2V_PATH = opt.w2v_path
    model.set_w2v_path(W2V_PATH)
    model.build_vocab_k_words(K=100000)

    train_dataset, valid_dataset, test_dataset = load_dataset_cap(opt=opt)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
    data_loaders = [train_loader, valid_loader, test_loader]

    embeddings_path = os.path.join(opt.data_path, 'sentence_embeddings.pkl')
    sentence_embeddings = load_pkl(embeddings_path)
    caption_set = get_caps(opt.data_path)

    cnt = 0

    for dataloader in data_loaders:
        for i, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, pos_feat, lens, gts, video_id) in enumerate(dataloader):
            for t in range(feats0.size(0)):
                vid_t = video_id[t]
                number = len(caption_set[vid_t])
                temp = []
                for j in range(number):
                    temp.append(caption_set[vid_t][j][b'tokenized'].decode())
                temp_embeddings = model.encode(temp, bsize=128, tokenize=False)
                keeped_embeddings = sentence_embeddings[vid_t]

                for j in range(number):
                    cosine_value = cosine(temp_embeddings[j], keeped_embeddings[j])
                    # print(cosine_value)
                    if math.fabs(1.0 - cosine_value) < EPS: continue
                    cnt += 1
            print('now cnt == ', cnt)

    print('cnt == ', cnt)