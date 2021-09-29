import os
import json
import h5py
import logging
import numpy as np
import pickle as pk

import torch
from torch import nn
import torch.nn.functional as F
from stanfordcorenlp import StanfordCoreNLP

from .utils import video2feats, moment_to_iou2d, embedding

class DidemoDataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, root, feat_file, tokenizer_folder, num_pre_clips,
                num_clips, pre_query_size, num_pyramid_layers=1):
        super(DidemoDataset, self).__init__()

        self.set_syntactic_relations()
        self.nlp = StanfordCoreNLP(tokenizer_folder)
        self.max_words = 0
        self.embedding = embedding()
        self.num_clips = num_clips
        self.num_pre_clips = num_pre_clips
        self.num_pyramid_layers = num_pyramid_layers

        #load annotation file
        annos = json.load(open(ann_file,'r'))

        cache = ann_file+'.pickle'
        if os.path.exists(cache):
            # if cached data exist load it.
            self._load_pickle_data(cache)
        else:
            # otherwise compute the annotations information
            self._compute_annotaions(annos, num_clips, cache, num_pyramid_layers)

        self.feats = video2feats(feat_file, annos['videos'].keys(), num_pre_clips, dataset_name="didemo")

    def __getitem__(self, idx):
        anno = self.annos[idx]
        vid = anno['vid']

        query, wordlen, dep = self._get_language_feature(anno)
        feat, iou2d = self._get_video_feature(anno, vid)
        return feat, query, wordlen, iou2d, idx, dep
    
    def __len__(self):
        return len(self.annos)
    
    def get_duration(self, idx):
        return self.annos[idx]['duration']
    
    def get_sentence(self, idx):
        return self.annos[idx]['sentence']
    
    def get_moment(self, idx):
        return self.annos[idx]['moment']
    
    def get_vid(self, idx):
        return self.annos[idx]['vid']
    
    def set_syntactic_relations(self):
        self.relations = {'det': 1, 'case': 2, 'punct': 3, 'nsubj': 4, 'obl': 5, 'obj': 6, 'conj': 7, 'amod': 8, 
             'cc': 9, 'advmod': 10, 'nmod': 11, 'compound': 12, 'mark': 13, 'dep': 14, 'nmod:poss': 15, 'xcomp': 16, 'aux:pass': 17, 
             'advcl': 18, 'nsubj:pass': 19, 'aux': 20, 'compound:prt': 21, 'acl':22, 'nummod': 23, 'ccomp': 24, 'cop': 25, 'fixed': 26, 
             'acl:relcl': 27, 'parataxis': 28, 'expl': 29, 'obl:tmod': 30, 'obl:npmod': 31, 'iobj': 32, 'det:predet': 33, 'appos': 34, 
             'csubj': 35, 'discourse': 36, 'csubj:pass': 37, 'cc:preconj': 38, 'orphan': 39, 'goeswith': 40, 'ROOT':41}

    def get_syntactic_relations(self):
        return self.relations

    def pad_dependency_matrix(self, l, dep):
        pad = (0, self.max_words-l, 0, self.max_words-l)
        return F.pad(input=dep, pad=pad, mode='constant', value=0)

    def concensus_among_annotators(self,timestamps):
        l = len(timestamps)
        ious = np.zeros((l,l-1))
        for i in range(l):
            t1 = np.asarray(timestamps[i],dtype='float32')
            cnt = 0
            for j in range(l):
                if i!= j:
                    t2 = np.asarray(timestamps[j],dtype='float32')
                    ious[i,cnt] = self.iou(t1,t2)
                    cnt +=1
        idx = ious.mean(axis=1).argmax()
        return timestamps[idx]

    def iou(self, t1, t2):
        inter = min(t1[1],t2[1]) - max(t1[0], t2[0])
        union = max(t1[1],t2[1]) - min(t1[0], t2[0])
        return max(inter,0.0) / union

    def _load_pickle_data(self,cache):
        '''
            The function loads preprocesses annotations and compute the max lenght of the sentences.

            INPUTS:
            cache: path to pickle file from where to load preprocessed annotations

            OUTPUTS:
            None.
        '''
        logger = logging.getLogger("vlg.trainer")
        logger.info("Load cache data, please wait...")

        self.annos = pk.load(open(cache, 'rb'))
        self.max_words = max([a['wordlen'] for a in self.annos])

    def _compute_annotaions(self, annos, num_clips, cache, num_pyramid_layers):
        '''
            The function processes the annotations computing language tokenizationa and query features.
            Construct the moment annotations for training and the target iou2d map.
            Processed the language to obtain syntactic dependencies.
            Dump everything in the pickle file for speading up following run.

            INPUTS:
            annos: annotations loaded from json files
            num_clips: number of clips (size of iou2d)
            cache: path to pickle file where to dump preprocessed annotations

            OUTPUTS:
            None.
        '''
        # compute the annotation data and dump it in a pickle file
        self.annos = []
        logger = logging.getLogger("vlg.trainer")
        logger.info("Preparing data, please wait...")
        videos = annos['videos']
        moments = annos['moments']
        for anno in moments:
            # Unpack Info ----------------------------------------------------------------
            vid = anno['video']
            duration  = float(videos[vid]['duration'])
            timestamp = self.concensus_among_annotators(anno['times'])
            sentence  = anno['description']
            standford_tokens = anno['syntactic_dependencies']['tokens']          
            syntactic_dep = torch.tensor(anno['syntactic_dependencies']['depencendy_matrices'])
            
            # Process gt annotations -----------------------------------------------------
            if timestamp[0] < timestamp[1]:
                moment = torch.tensor([max(timestamp[0], 0), min(timestamp[1], duration)])
            
            # Generate targets for training ----------------------------------------------
            iou2d = []
            for j in range(num_pyramid_layers):
                new_resolution = num_clips // (2 ** j)
                padding = nn.ZeroPad2d((0, num_clips - new_resolution, 
                                        0, num_clips - new_resolution))
                iou2d.append(padding(moment_to_iou2d(moment, new_resolution, duration)))
            iou2d = torch.stack(iou2d)

            # Language preprocessing-------------------------------------------------------
            tokens = self.nlp.word_tokenize(sentence)
            assert len(tokens)==len(standford_tokens)
            query = self.embedding(tokens)
            if len(tokens) > self.max_words:
                self.max_words = len(tokens)

            # Save preprocessed annotations ----------------------------------------------
            dump_dict = {
                'vid': vid,
                'moment': moment,
                'iou2d': iou2d,
                'sentence': sentence,
                'query': query,
                'wordlen': query.size(0),
                'duration': duration,
                'syntactic_dep': syntactic_dep,
            }

            self.annos.append(dump_dict)
        
        # save to file
        pk.dump(self.annos,open(cache,'wb'))

    def _get_language_feature(self, anno):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information

            OUTPUTS:
            query: features of the selected sentence
            wordlen: length of the selected sentence 
            dep: dependency matrix for the selected sentence (padded to max length for batch creation)
        '''
        query = anno['query']
        wordlen = anno['wordlen']
        syntactic_dep = anno['syntactic_dep']
        dep = self.pad_dependency_matrix(wordlen,syntactic_dep)
        return query, wordlen, dep

    def _get_video_feature(self, anno, vid):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            vid: video id to select the correct features

            OUTPUTS:
            feat: video features
            iou2d: target matrix 
        '''
        return self.feats[vid], anno['iou2d']
        