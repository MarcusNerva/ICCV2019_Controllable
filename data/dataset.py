from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import sys
sys.path.append('../')
import pickle
import numpy as np
import time
import myopts
from collections import OrderedDict
import random
import h5py
import numpy as np


def load_pkl(pkl_file):
    f = open(pkl_file, 'rb')
    try:
        ret = pickle.load(f, encoding='bytes')
    finally:
        f.close()
    return ret

def get_sub_frames(frames, K):
    if len(frames) < K:
        zeros = np.zeros([K - frames.shape[0], frames.shape[1]])
        ret = np.concatenate((frames, zeros), axis=0)
    else:
        idx = np.linspace(0, len(frames), K, endpoint=False, dtype=int)
        ret = frames[idx]
    return ret

def get_sub_pool_frames(frames, K):
    assert len(frames.shape) == 4, 'shape of pool features should be 4 dims'
    if len(frames) < K:
        dims = list(frames.shape)
        zeros = np.zeros([K - dims[0]] + dims[1:])
        ret = np.concatenate((frames, zeros), axis=0)
    else:
        idx = np.linspace(0, len(frames), K, endpoint=False, dtype=int)
        ret = frames[idx]
    return ret

def filt_word_category(cate_pkl, words):
    category_words = load_pkl(cate_pkl)
    words_category = {}
    for category, wordlist in category_words.items():
        for word in wordlist:
            words_category[word] = category

    category_name = {}
    category_name[1] = ['FW', '-LRB-', '-RRB-', 'LS']
    category_name[2] = ['VB', 'VBD', 'VBP', 'VBG', 'VBN', 'VBZ']
    category_name[3] = ['NN', 'NNS', 'NNP']
    category_name[4] = ['JJ', 'JJR', 'JJS']
    category_name[5] = ['RB', 'RBS', 'RBR', 'WRB', 'EX']
    category_name[6] = ['CC']
    category_name[7] = ['PRP', 'PRP$', 'WP', 'POS', 'WP$']
    category_name[8] = ['IN', 'TO']
    category_name[9] = ['DT', 'WDT', 'PDT']
    category_name[10] = ['RP', 'MD']
    category_name[11] = ['CD']
    category_name[12] = ['SYM', ':', '``', '#', '$']
    category_name[13] = ['UH']

    all_category = category_words.keys()
    category_id = {}
    for id, categorys in category_name.items():
        id = int(id)
        for cate in categorys:
            category_id[cate] = id

    all_words_in_category = words_category.keys()
    filted_words_cateid = {}
    for key in words:
        if key in all_words_in_category:
            cate = bytes.decode(words_category[key])
            id = category_id[cate]
            filted_words_cateid[key] = id
        else:
            filted_words_cateid[key] = 1
    # print('now cate\'s type is ', type(cate))
    filted_words_cateid['<EOS>'] = 0
    filted_words_cateid['<UNK>'] = 1
    unmasked_categoryid = [i for i in range(14)]
    return filted_words_cateid, words_category, category_id, category_words, unmasked_categoryid

def get_caps(path):
    return load_pkl(os.path.join(path, 'CAP.pkl'))

def get_nwords(path):
    return len(load_pkl(os.path.join(path, 'worddict.pkl'))) + 2

def get_nclasses(path):
    return 14

def get_itow(data_path):
    ret = {}
    path = os.path.join(data_path, 'worddict.pkl')
    wtoi = load_pkl(path)
    wtoi[b'<EOS>'] = 0
    wtoi[b'<UNK>'] = 1
    for word, id in wtoi.items():
        ret[id] = word.decode()
    return ret

class Dset_train(Dataset):
    # def get_itow(self):
    #     wtoi = self.wtoi
    #     itow = {}
    #     for word, id in wtoi.iteritems():
    #         itow[id] = word
    #     return itow

    def __init__(self, train_pkl, cap_pkl, cate_pkl, feat0_store, feat1_store, wtoi_path, pos_store=None, nwords=10000, K=28, opt=None):
        super(Dset_train, self).__init__()
        self.nwords = nwords
        self.K = K
        data_name_list = load_pkl(train_pkl)
        caps = load_pkl(cap_pkl)
        wtoi = load_pkl(wtoi_path)
        wtoi['<EOS>'] = 0
        wtoi['<UNK>'] = 1
        wtoi_keys = wtoi.keys()
        self.wtoi = wtoi
        filt_word_cateid, word_cate, cate_id, cate_word, unmasked_cateid = filt_word_category(cate_pkl, wtoi)
        self.category = filt_word_cateid
        cate_keys = self.category.keys()

        temp_cap_list = []
        # print(data_name_list[0].decode())
        for i, ID in enumerate(data_name_list):
            vidid, capid = ID.decode().split('_')
            vidid = vidid.encode()
            temp_cap_list.append(caps[vidid][int(capid)])

        # 将bytes类型[b'']看作是被加密的类型, bytes->str需要解密[decode()], 而str->bytes需要加密[encode()]

        data_list = []
        cap_list = []
        for data, cap in zip(data_name_list, temp_cap_list):
            token = cap[b'tokenized'].split()
            # split()对于bytes 类型的字符串同样适用
            if 0 < len(token) and len(token) <= opt.seq_length:
                data_list.append(data)
                new_cap = {}
                new_cap['caption'] = cap[b'caption']
                new_cap['tokenized'] = cap[b'tokenized']
                new_cap['numbered'] = [ wtoi[w] if w in wtoi_keys else 1 for w in token ]
                new_cap['category'] = [ self.category[w] if w in cate_keys else 1 for w in token ]
                new_cap['category_mask'] = [1 if idx in unmasked_cateid else 0 for idx in new_cap['category']]
                cap_list.append(new_cap)

        gts_list = []
        for i, ID in enumerate(data_list):
            sub_gts_list = []
            vidid, _ = ID.decode().split('_')
            vidid = vidid.encode()
            for cap in caps[vidid]:
                token = cap[b'tokenized'].split()
                numbered = [ wtoi[w] if w in wtoi_keys else 1 for w in token ]
                sub_gts_list.append(numbered)
            sub_gts_list.sort(key=lambda x: len(x), reverse=True)
            tmp_gts_arr = np.zeros([len(sub_gts_list), len(sub_gts_list[0])], dtype=int)
            for x in range(len(sub_gts_list)):
                tmp_gts_arr[x, :len(sub_gts_list[x])] = sub_gts_list[x]
            gts_list.append(tmp_gts_arr)

        self.data_list = data_list
        self.cap_list = cap_list
        self.gts_list = gts_list
        self.feat0_store = feat0_store
        self.feat1_store = feat1_store
        self.pos_store = pos_store

    def __len__(self):
        return len(self.cap_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        cap = self.cap_list[idx]['numbered']
        cap_class = self.cap_list[idx]['category']
        class_mask = self.cap_list[idx]['category_mask']
        gts = self.gts_list[idx]

        feat0 = self.feat0_store[data.decode().split('_')[0]][:]
        feat0 = get_sub_frames(feat0, self.K)
        feat0 = torch.from_numpy(feat0).float()

        feat1 = self.feat1_store[data.decode().split('_')[0]][:]
        feat1 = get_sub_frames(feat1, self.K)
        feat1 = torch.from_numpy(feat1).float()

        if self.pos_store is not None:
            pos_feat = self.pos_store[data.decode().split('_')[0]]['states'][:]
            pos_feat = pos_feat[-1]
            pos_feat = torch.from_numpy(pos_feat).float()
        else:
            pos_feat = None

        feat_mask = (torch.sum(feat0.view(feat0.size(0), -1), dim=1, keepdim=True) != 0).float().transpose(1, 0)

        if self.pos_store is None:
            return data, cap, cap_class, class_mask, feat0, feat1, feat_mask, gts
        return data, cap, cap_class, class_mask, feat0, feat1, feat_mask, pos_feat, gts

class Dset_test(Dataset):
    # def get_itow(self):
    #     wtoi = self.wtoi
    #     itow = {}
    #     for word, id in wtoi.iteritems():
    #         itow[id] = word
    #     return itow

    def __init__(self, test_pkl, cap_pkl, cate_pkl, feat0_store, feat1_store, wtoi_path, pos_store=None, nwords=10000, K=28, opt=None):
        super(Dset_test, self).__init__()
        self.nwords = nwords
        self.K = K
        data_name_list = load_pkl(test_pkl)
        caps = load_pkl(cap_pkl)
        wtoi = load_pkl(wtoi_path)
        wtoi['<EOS>'] = 0
        wtoi['<UNK>'] = 1
        wtoi_keys = wtoi.keys()
        self.wtoi = wtoi
        filt_word_cateid, word_cate, cate_id, cate_word, unmasked_cateid = filt_word_category(cate_pkl, wtoi)
        self.category = filt_word_cateid
        cate_keys = self.category.keys()

        temp_cap_list = []
        # print(data_name_list[0].decode())
        for i, ID in enumerate(data_name_list):
            vidid, capid = ID.decode().split('_')
            vidid = vidid.encode()
            temp_cap_list.append(caps[vidid][int(capid)])

        data_list = []
        cap_list = []
        for data, cap in zip(data_name_list, temp_cap_list):
            token = cap[b'tokenized'].split()
            # print('token is ', token, 'and token\'s type is ', type(token))
            if 0 < len(token) and len(token) <= opt.seq_length:
                data_list.append(data)
                new_cap = {}
                new_cap['caption'] = cap[b'caption']
                new_cap['tokenized'] = cap[b'tokenized']
                new_cap['numbered'] = [ wtoi[w] if w in wtoi_keys else 1 for w in token ]
                new_cap['category'] = [ self.category[w] if w in cate_keys else 1 for w in token ]
                new_cap['category_mask'] = [1 if idx in unmasked_cateid else 0 for idx in new_cap['category']]
                cap_list.append(new_cap)

        tmp_vid = []
        tmp_vidname = []
        tmp_cap = []
        for data, cap in zip(data_list, cap_list):
            if data.decode().split('_')[0] not in tmp_vid:
                tmp_vid.append(data.decode().split('_')[0])
                tmp_vidname.append(data)
                tmp_cap.append(cap)
        data_list = tmp_vidname
        cap_list = tmp_cap

        gts_list = []
        for i, ID in enumerate(data_list):
            sub_gts_list = []
            vidid, _ = ID.decode().split('_')
            vidid = vidid.encode()
            for cap in caps[vidid]:
                token = cap[b'tokenized'].split()
                numbered = [ wtoi[w] if w in wtoi_keys else 1 for w in token ]
                sub_gts_list.append(numbered)
            sub_gts_list.sort(key=lambda x: len(x), reverse=True)
            tmp_gts_arr = np.zeros([len(sub_gts_list), len(sub_gts_list[0])], dtype=int)
            for x in range(len(sub_gts_list)):
                tmp_gts_arr[x, :len(sub_gts_list[x])] = sub_gts_list[x]
            gts_list.append(tmp_gts_arr)

        self.data_list = data_list
        self.cap_list = cap_list
        self.gts_list = gts_list
        self.feat0_store = feat0_store
        self.feat1_store = feat1_store
        self.pos_store = pos_store

    def __len__(self):
        return len(self.cap_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        cap = self.cap_list[idx]['numbered']
        cap_class = self.cap_list[idx]['category']
        class_mask = self.cap_list[idx]['category_mask']
        gts = self.gts_list[idx]

        # print("data.split is ", data.split('_')[0])
        # keys = self.feat0_store.keys()
        # print([key for key in keys])
        # print([type(key) for key in keys])
        feat0 = self.feat0_store[data.decode().split('_')[0]][:]
        feat0 = get_sub_frames(feat0, self.K)
        feat0 = torch.from_numpy(feat0).float()

        feat1 = self.feat1_store[data.decode().split('_')[0]][:]
        feat1 = get_sub_frames(feat1, self.K)
        feat1 = torch.from_numpy(feat1).float()

        feat_mask = (torch.sum(feat0.view(feat0.size(0), -1), dim=1, keepdim=True) != 0).float().transpose(1, 0)
        # print('$$$$$$$$feat_mask.shape is ', feat_mask.shape)

        if self.pos_store is not None:
            pos_feat = self.pos_store[data.decode().split('_')[0]]['states'][:]
            pos_feat = pos_feat[-1]
            pos_feat = torch.from_numpy(pos_feat).float()
        else:
            pos_feat = None

        if self.pos_store is not None:
            return data, cap, cap_class, class_mask, feat0, feat1, feat_mask, pos_feat, gts
        return data, cap, cap_class, class_mask, feat0, feat1, feat_mask, gts

def collate_fn_pos(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    data, cap, cap_class, class_mask, feat1, feat2, feat_mask, gts = zip(*batch)

    max_len = len(cap[0])
    feats0 = torch.stack(feat1, dim=0)
    feats1 = torch.stack(feat2, dim=0)
    feat_mask = torch.cat(feat_mask, dim=0)

    caps = []
    lens = []
    caps_mask = torch.zeros([len(cap), max_len + 1])
    for i in range(len(cap)):
        temp_cap = [0] * (max_len + 1)
        temp_cap[1:len(cap[i]) + 1] = cap[i]
        caps.append(temp_cap)
        caps_mask[i, :len(cap[i]) + 1] = 1
        lens.append(len(cap[i]))
    caps = torch.LongTensor(caps)

    cap_classes = []
    class_masks = []

    for i in range(len(cap_class)):
        temp_cap_class = [0] * (max_len + 1)
        temp_cap_class[0:len(cap_class[i])] = cap_class[i]
        cap_classes.append(temp_cap_class)
        temp_class_mask = [0] * (max_len + 1)
        temp_class_mask[0:len(class_mask[i])] = class_mask[i]
        temp_class_mask[len(class_mask[i])] = 1
        class_masks.append(temp_class_mask)

    cap_classes = torch.LongTensor(cap_classes)
    class_masks = torch.FloatTensor(class_masks)

    gts = [torch.from_numpy(x).long() for x in gts]
    video_id = [i.decode().split('_')[0] for i in data]

    return data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, lens, gts, video_id

def collate_fn_cap(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    data, cap, cap_class, class_mask, feats0, feats1, feat_mask, pos_feat, gts = zip(*batch)

    max_len = len(cap[0])
    feats0 = torch.stack(feats0, dim=0)
    feats1 = torch.stack(feats1, dim=0)
    feat_mask = torch.cat(feat_mask, dim=0)
    pos_feat = torch.stack(pos_feat, dim=0)

    caps = []
    lens = []
    caps_mask = torch.zeros([len(cap), max_len + 1])
    for i in range(len(cap)):
        temp_cap = [0] * (max_len + 1)
        temp_cap[1:len(cap[i]) + 1] = cap[i]
        caps.append(temp_cap)
        caps_mask[i, :len(cap[i]) + 1] = 1
        lens.append(len(cap[i]))
    caps = torch.LongTensor(caps)

    cap_classes = []
    class_masks = []
    for i in range(len(cap_class)):
        temp_cap_class = [0] * (max_len + 1)
        temp_cap_class[0:len(cap_class[i])] = cap_class[i]
        cap_classes.append(temp_cap_class)
        temp_class_mask = [0] * (max_len + 1)
        temp_class_mask[0:len(class_mask[i])] = class_mask[i]
        temp_class_mask[len(class_mask[i])] = 1
        class_masks.append(temp_class_mask)

    cap_classes = torch.LongTensor(cap_classes)
    class_masks = torch.FloatTensor(class_masks)

    gts = [torch.from_numpy(x).long() for x in gts]
    video_id = [i.decode().split('_')[0] for i in data]

    return data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, pos_feat, lens, gts, video_id

def load_dataset_pos(opt):
    train_pkl = os.path.join(opt.data_path, 'train.pkl')
    valid_pkl = os.path.join(opt.data_path, 'valid.pkl')
    test_pkl = os.path.join(opt.data_path, 'test.pkl')
    cap_pkl = os.path.join(opt.data_path, 'CAP.pkl')
    cate_pkl = os.path.join(opt.data_path, 'category.pkl')
    feat0_path = os.path.join(opt.data_path, 'IR_feats.hdf5')
    feat1_path = os.path.join(opt.data_path, 'I3D_feats.hdf5')
    wtoi_path = os.path.join(opt.data_path, 'worddict.pkl')
    feat0 = h5py.File(feat0_path, 'r')
    feat1 = h5py.File(feat1_path, 'r')

    train_dataset = Dset_train(train_pkl=train_pkl, cap_pkl=cap_pkl, cate_pkl=cate_pkl, feat0_store=feat0, feat1_store=feat1, wtoi_path=wtoi_path, nwords=opt.vocab_size, K=opt.seq_length, opt=opt)
    valid_dataset = Dset_test(test_pkl=valid_pkl, cap_pkl=cap_pkl, cate_pkl=cate_pkl, feat0_store=feat0, feat1_store=feat1, wtoi_path=wtoi_path, nwords=opt.vocab_size, K=opt.seq_length, opt=opt)
    test_dataset = Dset_test(test_pkl=test_pkl, cap_pkl=cap_pkl, cate_pkl=cate_pkl, feat0_store=feat0, feat1_store=feat1, wtoi_path=wtoi_path, nwords=opt.vocab_size, K=opt.seq_length, opt=opt)
    return train_dataset, valid_dataset, test_dataset

def load_dataset_cap(opt):
    train_pkl = os.path.join(opt.data_path, 'train.pkl')
    valid_pkl = os.path.join(opt.data_path, 'valid.pkl')
    test_pkl = os.path.join(opt.data_path, 'test.pkl')
    cap_pkl = os.path.join(opt.data_path, 'CAP.pkl')
    cate_pkl = os.path.join(opt.data_path, 'category.pkl')
    feat0_path = os.path.join(opt.data_path, 'IR_feats.hdf5')
    feat1_path = os.path.join(opt.data_path, 'I3D_feats.hdf5')
    wtoi_path = os.path.join(opt.data_path, 'worddict.pkl')
    # pos_train_path = os.path.join(opt.data_path, 'pos_features/train.hdf5')
    # pos_valid_path = os.path.join(opt.data_path, 'pos_features/valid.hdf5')
    # pos_test_path = os.path.join(opt.data_path, 'pos_features/test.hdf5')

    pos_train_path = os.path.join(opt.data_path, 'postagsequence.hdf5')
    pos_valid_path = os.path.join(opt.data_path, 'postagsequence.hdf5')
    pos_test_path = os.path.join(opt.data_path, 'postagsequence.hdf5')

    feat0 = h5py.File(feat0_path, 'r')
    feat1 = h5py.File(feat1_path, 'r')
    pos_train_feature = h5py.File(pos_train_path, 'r')
    pos_valid_feature = h5py.File(pos_valid_path, 'r')
    pos_test_feature = h5py.File(pos_test_path, 'r')

    train_dataset = Dset_train(train_pkl=train_pkl, cap_pkl=cap_pkl, cate_pkl=cate_pkl, feat0_store=feat0, feat1_store=feat1, wtoi_path=wtoi_path, pos_store=pos_train_feature, nwords=opt.vocab_size, K=opt.seq_length, opt=opt)
    valid_dataset = Dset_test(test_pkl=valid_pkl, cap_pkl=cap_pkl, cate_pkl=cate_pkl, feat0_store=feat0, feat1_store=feat1, wtoi_path=wtoi_path, pos_store=pos_valid_feature, nwords=opt.vocab_size, K=opt.seq_length, opt=opt)
    test_dataset = Dset_test(test_pkl=test_pkl, cap_pkl=cap_pkl, cate_pkl=cate_pkl, feat0_store=feat0, feat1_store=feat1, wtoi_path=wtoi_path, pos_store=pos_test_feature, nwords=opt.vocab_size, K=opt.seq_length, opt=opt)
    return train_dataset, valid_dataset, test_dataset

class Opt_stub:
    def __init__(self):
        super(Opt_stub, self).__init__()
        self.seq_length = 28
        self.feat_K = 28
        self.batch_size = 3
        self.vocab_size = 20000
        self.data_path = '/Users/bismarck/PycharmProjects/ICCV2019_Controllable/data'

if __name__=='__main__':
    opt = Opt_stub()
    # print('n_words is ', get_nwords(opt.data_path))
    # pos_train_dataset, pos_valid_dataset, pos_test_dataset = load_dataset_pos(opt=opt)
    cap_train_dataset, cap_valid_dataset, cap_test_dataset = load_dataset_cap(opt=opt)
#
    # pos_trainloader = DataLoader(pos_train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_pos)
    # pos_validloader = DataLoader(pos_valid_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_pos)
    # pos_testloader = DataLoader(pos_test_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_pos)
#
#     cap_trainloader = DataLoader(cap_train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
#     cap_validloader = DataLoader(cap_valid_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
#     cap_testloader = DataLoader(cap_test_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)

    # print("=================now is training=====================")
    # for i, (data, caps, caps_mask, cap_classes, class_masks, feats1, feats2, feat_mask, lens, gts, image_id) in enumerate(pos_trainloader):
    #     if i % 100 == 0:
    #         print('data.shape == ', np.array(data).shape)
    #         print('caps.shape == ', np.array(caps).shape)
    #         print('caps_mask.shape == ', np.array(caps_mask).shape)
    #         print('cap_classes.shape == ', np.array(cap_classes).shape)
    #         print('feats1.shape == ', np.array(feats1).shape)
    #         print('feats2.shape == ', np.array(feats2).shape)
    #         print('feat_mask.shape == ', np.array(feat_mask).shape)
    #         print('lens.shape == ', np.array(lens).shape)
    #         # print('gts.shape == ', np.array(gts).shape)
    #         print('image_id.shape == ', np.array(image_id).shape)

    # print("=================now is valid=====================")
    # for i, (data, caps, caps_mask, cap_classes, class_masks, feats1, feats2, feat_mask, lens, gts, image_id) in enumerate(pos_validloader):
    #     if i % 100 == 0:
    #         print('data.shape == ', np.array(data).shape)
    #         print('caps.shape == ', np.array(caps).shape)
    #         print('caps_mask.shape == ', np.array(caps_mask).shape)
    #         # print('cap_classes.shape == ', np.array(cap_classes).shape)
    #         # print(cap_classes)
    #         print(class_masks)
    #         print('class_masks.data[0, :] == ', class_masks.data[0, :])
    #         print(np.argwhere(class_masks.data[0, :] != 0))
    #         print('class_masks.shape == ', class_masks.shape)
    #         print('feats1.shape == ', np.array(feats1).shape)
    #         print('feats2.shape == ', np.array(feats2).shape)
    #         print('feat_mask.shape == ', np.array(feat_mask).shape)
    #         print('lens.shape == ', np.array(lens).shape)
    #         # print('gts.shape == ', np.array(gts).shape)
    #         print(gts)
    #         print('image_id.shape == ', np.array(image_id).shape)
    #
    # print("=================now is testing=====================")
    # for i, (data, caps, caps_mask, cap_classes, class_masks, feats1, feats2, feat_mask, lens, gts, image_id) in enumerate(pos_testloader):
    #     if i % 100 == 0:
    #         print('data.shape == ', np.array(data).shape)
    #         print('caps.shape == ', np.array(caps).shape)
    #         print('caps_mask.shape == ', np.array(caps_mask).shape)
    #         print('cap_classes.shape == ', np.array(cap_classes).shape)
    #         print('feats1.shape == ', np.array(feats1).shape)
    #         print('feats2.shape == ', np.array(feats2).shape)
    #         print('feat_mask.shape == ', np.array(feat_mask).shape)
    #         print('lens.shape == ', np.array(lens).shape)
    #         # print('gts.shape == ', np.array(gts).shape)
    #         print('image_id.shape == ', np.array(image_id).shape)

    # print("=================now is training=====================")
    # for i, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, pos_feat, lens, gts, video_id) in enumerate(cap_trainloader):
    #     if i % 100 == 0:
    #         print('data.shape == ', np.array(data).shape)
    #         print('caps.shape == ', np.array(caps).shape)
    #         print('caps_mask.shape == ', np.array(caps_mask).shape)
    #         print('cap_classes.shape == ', np.array(cap_classes).shape)
    #         print('feats1.shape == ', np.array(feats0).shape)
    #         print('feats2.shape == ', np.array(feats1).shape)
    #         print('feat_mask.shape == ', np.array(feat_mask).shape)
    #         print('lens.shape == ', np.array(lens).shape)
    #         # print('gts.shape == ', np.array(gts).shape)
    #         print('image_id.shape == ', np.array(video_id).shape)

    # print("=================now is valid=====================")
    # for i, (
    # data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, pos_feat, lens, gts, video_id) in enumerate(
    #         cap_validloader):
    #     if i % 100 == 0:
    #         print('data.shape == ', np.array(data).shape)
    #         print('caps.shape == ', np.array(caps).shape)
    #         print('caps_mask.shape == ', np.array(caps_mask).shape)
    #         print('cap_classes.shape == ', np.array(cap_classes).shape)
    #         print('feats1.shape == ', np.array(feats0).shape)
    #         print('feats2.shape == ', np.array(feats1).shape)
    #         print('feat_mask.shape == ', np.array(feat_mask).shape)
    #         print('lens.shape == ', np.array(lens).shape)
    #         # print('gts.shape == ', np.array(gts).shape)
    #         print('image_id.shape == ', np.array(video_id).shape)
    #
    # print("=================now is testing=====================")
    # for i, (
    # data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, pos_feat, lens, gts, video_id) in enumerate(
    #         cap_testloader):
    #     if i % 100 == 0:
    #         print('data.shape == ', np.array(data).shape)
    #         print('caps.shape == ', np.array(caps).shape)
    #         print('caps_mask.shape == ', np.array(caps_mask).shape)
    #         print('cap_classes.shape == ', np.array(cap_classes).shape)
    #         print('feats1.shape == ', np.array(feats0).shape)
    #         print('feats2.shape == ', np.array(feats1).shape)
    #         print('feat_mask.shape == ', np.array(feat_mask).shape)
    #         print('lens.shape == ', np.array(lens).shape)
    #         # print('gts.shape == ', np.array(gts).shape)
    #         print('image_id.shape == ', np.array(video_id).shape)