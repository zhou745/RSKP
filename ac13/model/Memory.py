import torch
import torch.nn as nn
import torch.nn.init as torch_init
import random
import numpy as np


class Memory(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_mu = args.n_mu
        self.n_class = args.action_cls_num
        self.n_video = args.video_num
        self.n_sample = args.sample_num
        self.out_dim = args.out_feat_num

        self.momentum = args.momentum

        self.label_to_indices = {}

        self.register_buffer("queue", torch.zeros(self.n_video, self.n_mu, self.out_dim))
        torch_init.xavier_uniform_(self.queue)

    @torch.no_grad()
    def _update_queue(self, inp, vid_idx):
        self.queue[vid_idx, ...] = self.queue[vid_idx, ...] * self.momentum + inp * (1 - self.momentum)

    def _init_queue(self, ft_queue, label_queue):
        # initialize mu queue
        self.queue = ft_queue.detach()
        # initialize label queue
        for label in range(self.n_class):
            label_select = label_queue[:, label]
            self.label_to_indices[label] = np.where(label_select == 1)[0]

    def _return_queue(self, vid_idxs, labels):
        batch_select_vids = []
        for i in range(len(vid_idxs)):
            vid_idx = vid_idxs[i]
            label = labels[i]
            cls_idx = np.where(label == 1)[0][0]
            select_idx = random.sample([ele for ele in self.label_to_indices[cls_idx] if ele != vid_idx], self.n_sample)
            select_vid = self.queue[select_idx]
            select_vid = select_vid.view(1, -1, select_vid.size(2))
            batch_select_vids.append(select_vid)

        return torch.cat(batch_select_vids, 0)