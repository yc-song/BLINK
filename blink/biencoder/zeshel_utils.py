# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Utility code for zeshel dataset
import json
import torch

DOC_PATH = "data/zeshel/documents/"
WORLDS = [
    'american_football',
    'doctor_who',
    'fallout',
    'final_fantasy',
    'military',
    'pro_wrestling',
    'starwars',
    'world_of_warcraft',
    'coronation_street',
    'muppets',
    'ice_hockey',
    'elder_scrolls',
    'forgotten_realms',
    'lego',
    'star_trek',
    'yugioh'
]

world_to_id = {src : k for k, src in enumerate(WORLDS)}


def load_entity_dict_zeshel(logger, params):
    entity_dict = {}
    # different worlds in train/valid/test
    if params["mode"] == "train":
        start_idx = 0
        end_idx = 8
    elif params["mode"] == "valid":
        start_idx = 8
        end_idx = 12
    else:
        start_idx = 12
        end_idx = 16
    # load data
    for i, src in enumerate(WORLDS[start_idx:end_idx]):
        fname = DOC_PATH + src + ".json"
        cur_dict = {}
        doc_list = []
        src_id = world_to_id[src]
        with open(fname, 'rt') as f:
            for line in f:
                sample = json.loads(line.rstrip())
                title = sample['title']
                text = sample.get("text", "").strip()
                doc_list.append((title, text))
                if params["debug"] and len(doc_list) > 200:
                    break
        logger.info("Load for world %s." % src)
        entity_dict[src_id] = doc_list
    return entity_dict


class Stats():
    def __init__(self, top_k=1000):
        self.cnt = 0
        self.rank = [1, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1000]
        self.LEN = len(self.rank) 
        self.hits = [0]*self.LEN
        if top_k == -1:
            self.top_k = 1000
        else:
            self.top_k = top_k


    def add(self, idx):
        self.cnt += 1
        if idx == -1:
            return
        for i in range(self.LEN):
            if idx < self.rank[i]:
                self.hits[i] += 1

    def extend(self, stats):
        self.cnt += stats.cnt
        for i in range(self.LEN):
            self.hits[i] += stats.hits[i]

    def output(self):
        output_json = "Total: %d examples." % self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))
        return output_json

