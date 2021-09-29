import os
from os.path import join, exists
import h5py
import numpy as np

import torch
import torchtext
from torch.functional import F

import spacy
from spellchecker  import SpellChecker

from random import randint

def iou(candidates, gt):
    start, end = candidates[:,0].float(), candidates[:,1].float()
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = torch.nonzero(score2d, as_tuple=False)   
    scores = score2d[grids[:,0], grids[:,1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores

def moment_to_iou2d(moment, num_clips, duration):
    iou2d = torch.ones(num_clips, num_clips)
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d


def avgfeats(feats, num_pre_clips):
    # # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    # num_src_clips = feats.size(0)
    # idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    # idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # # To prevent a empty selection, check the idxs
    # meanfeats = []
    # for i in range(num_pre_clips):
    #     s, e = idxs[i], idxs[i+1]
    #     if s < e:
    #         meanfeats.append(feats[s:e].mean(dim=0))
    #     else:
    #         meanfeats.append(feats[s])
    output = F.interpolate(feats.transpose(0,1).unsqueeze(0), size=num_pre_clips, mode='linear', align_corners=False)
    return  output[0,...].transpose(0,1)

def video2feats(feat_file, vids, num_pre_clips, dataset_name):
    assert exists(feat_file), '{} not found'.format(feat_file)
    vid_feats = {}
    with h5py.File(feat_file, 'r') as f:
        for vid in vids:
            if dataset_name == "activitynet":
                feat = f[vid]['c3d_features'][:]
            else:
                feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat),dim=1)
            interpolated_feat = avgfeats(feat, num_pre_clips) # feat
            vid_feats[vid] = interpolated_feat
    return vid_feats

class embedding(object):
    def __init__(self, vocabs=[], embedders=[], nlp=None, spell=None):
         # Glove setup
        self.vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
        self.vocab.itos.extend(['<unk>'])
        self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
        self.vocab.vectors = torch.cat(
            [self.vocab.vectors, torch.zeros(1, self.vocab.dim)],
            dim=0
        )
        # Spacy setup
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        # setup spellchecker
        self.spell = SpellChecker()
        self.spell.distance = 1                       ## Too expensive
        self.spell.word_frequency.load_words(list(self.vocab.stoi.keys()))
        self.embedder = torch.nn.Embedding.from_pretrained(self.vocab.vectors)
        self.cnt = 0

    def __call__(self, tokens):
        # Look up tokens in Glove vocabulary
        tokens_idx = []
        for t in tokens:
            t = t.lower()
            i = self.vocab.stoi.get(t, 400000) 
            if i == 400000:
                #try lemmatization
                t_ = self.nlp(t)[0]
                i = self.vocab.stoi.get(t_.lemma_, 400000)
                if  i == 400000:
                    # try spelling correction
                    t_ = self.spell.correction(t)
                    i  = self.vocab.stoi.get(t_, 400000)
                    if i == 400000:
                        self.cnt += 1
            tokens_idx.append(i)
            
        word_idxs = torch.tensor(tokens_idx, dtype=torch.long)
        return self.embedder(word_idxs)
   
