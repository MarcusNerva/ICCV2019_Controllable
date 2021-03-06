import torch
import numpy as np
import os
import sys
sys.path.append('../')
from infersent_model import InferSent
import myopts
import pickle
from eval.eval_cap import cosine
from collections import OrderedDict

def test(opt, infermodel, embed):
    import math
    EPS = 1e-4
    # path0 = os.path.join(opt.data_path, 'test_prediction.pkl')
    path1 = os.path.join(opt.data_path, 'extracted_test_prediction.pkl')
    # with open(path0, 'rb') as f:
    #     content0 = pickle.load(f)
    with open(path1, 'rb') as f:
        content1 = pickle.load(f)
    store = []
    for key in content1:
        store.append(content1[key])
    embeddings = infermodel.encode(store, bsize=128, tokenize=True)

    idx = 0
    for key in content1:
        sent = content1[key]
        temp = infermodel.encode([sent], bsize=128, tokenize=True)[0]
        if math.fabs(1 - cosine(temp, embeddings[idx])) > EPS:
            print(key)
            print(sent)
            print(cosine(temp, embeddings[idx]))
        idx += 1

    # for key in content0:
    #     sentence0 = content0[key].strip()
    #     sentence1 = content1[key].strip()
    #     if sentence0 == sentence1:
    #         print(key)
    #         print(sentence0)
    #         print(sentence1)


if __name__ == '__main__':
    opt = myopts.parse_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path0 = os.path.join(opt.data_path, 'whole.pkl')
    path1 = os.path.join(opt.data_path, 'without_pos.pkl')
    path2 = os.path.join(opt.data_path, 'without_XGate.pkl')
    path3 = os.path.join(opt.data_path, 'without_reward.pkl')
    path4 = os.path.join(opt.data_path, 'without_semantics.pkl')
    total_embeddings_path = os.path.join(opt.data_path, 'sentence_embeddings.pkl')
    # test(opt=opt)

    with open(path0, 'rb') as f:
        content0 = pickle.load(f)
    with open(path1, 'rb') as f:
        content1 = pickle.load(f)
    with open(path2, 'rb') as f:
        content2 = pickle.load(f)
    with open(path3, 'rb') as f:
        content3 = pickle.load(f)
    with open(path4, 'rb') as f:
        content4 = pickle.load(f)
    with open(total_embeddings_path, 'rb') as f:
        total_embeddings = pickle.load(f)

    model_version = 2
    MODEL_PATH = opt.infersent_model_path
    assert MODEL_PATH is not None, '--infersent_model_path is None!'
    MODEL_PATH = os.path.join(MODEL_PATH, 'infersent%s.pkl' % model_version)
    params_model = {
        'bsize': 64,
        'word_emb_dim': 300,
        'enc_lstm_dim': 2048,
        'pool_type': 'max',
        'dpout_model': 0.0,
        'version': model_version
    }
    infersent_model = InferSent(params_model)
    infersent_model.load_state_dict(torch.load(MODEL_PATH))
    infersent_model = infersent_model.to(device)
    W2V_PATH = opt.w2v_path
    assert W2V_PATH is not None, '--w2v_path is None!'
    infersent_model.set_w2v_path(W2V_PATH)
    infersent_model.build_vocab_k_words(K=100000)

    scores0, scores1, scores2, scores3, scores4 = [], [], [], [], []
    sentences0, sentences1, sentences2, sentences3, sentences4 = [], [], [], [], []
    compare01, compare02, compare03, compare04 = [], [], [], []
    keys = []

    for item in content0:
        keys.append(item)

    keys = sorted(keys, key=lambda x: int(x[3:]))

    for item in keys:
        # print(item)
        sentences0.append(content0[item])
        sentences1.append(content1[item])
        sentences2.append(content2[item])
        sentences3.append(content3[item])
        sentences4.append(content4[item])

    embeddings0 = infersent_model.encode(sentences0, bsize=128, tokenize=True)
    embeddings1 = infersent_model.encode(sentences1, bsize=128, tokenize=True)
    embeddings2 = infersent_model.encode(sentences2, bsize=128, tokenize=True)
    embeddings3 = infersent_model.encode(sentences3, bsize=128, tokenize=True)
    embeddings4 = infersent_model.encode(sentences4, bsize=128, tokenize=True)

    test(opt=opt, infermodel=infersent_model, embed=embeddings0)

    print(len(embeddings0))
    for i in range(7010, len(embeddings0)):
        vid = 'vid' + str(i + 1)
        b_vid = vid.encode()
        answers = total_embeddings[b_vid]
        max_score0, max_score1, max_score2, max_score3, max_score4 = -1.0, -1.0, -1.0, -1.0, -1.0
        # print('len of answers is ', len(answers))
        for item in answers:
            max_score0 = max(max_score0, cosine(item, embeddings0[i]))
            max_score1 = max(max_score1, cosine(item, embeddings1[i]))
            max_score2 = max(max_score2, cosine(item, embeddings2[i]))
            max_score3 = max(max_score3, cosine(item, embeddings3[i]))
            max_score4 = max(max_score4, cosine(item, embeddings4[i]))
        scores0.append(max_score0)
        scores1.append(max_score1)
        scores2.append(max_score2)
        scores3.append(max_score3)
        scores4.append(max_score4)

        if max_score0 > max_score1: compare01.append(vid)
        if max_score0 > max_score2: compare02.append(vid)
        if max_score0 > max_score3: compare03.append(vid)
        if max_score0 > max_score4: compare04.append(vid)

    compare_store = OrderedDict()
    compare_store['compare01'] = compare01
    compare_store['compare02'] = compare02
    compare_store['compare03'] = compare03
    compare_store['compare04'] = compare04

    print('mean_score0 == ', np.array(scores0).mean())
    print('mean_score1 == ', np.array(scores1).mean())
    print('mean_score2 == ', np.array(scores2).mean())
    print('mean_score3 == ', np.array(scores3).mean())
    print('mean_score4 == ', np.array(scores4).mean())

    print('len(compare01) == ', len(compare01))
    print('len(compare02) == ', len(compare02))
    print('len(compare03) == ', len(compare03))
    print('len(compare04) == ', len(compare04))

    store_path = os.path.join(opt.data_path, 'compare_store.pkl')
    with open(store_path, 'wb') as f:
        pickle.dump(compare_store, f)
