import os
import random
import warnings

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchaudio import transforms

from extract_features import get_mfcc

# root_dir = "/run/media/dallasautumn/data/duan-qiu-yang/2019-2020-现代程序设计技术-大作业-数据/A类问题/A3-闻声知乐/CASIA情感语料库/CASIA情感语料库"
# print(os.listdir(base_dir))


class AudioDataset(Dataset):
    """
    定义音频情感数据集
    从testandtrain中抽取
    Thanks for the previous work of Weng Di :)
    """

    def __init__(self, root="./datasets", train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train  # "train" or "test"
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.filedir = os.path.join(self.root, "train")
        else:
            self.filedir = os.path.join(self.root, "test")

        self.data = [self.get_data(filename)
                     for filename in os.listdir(self.filedir)]
        self.targets = [self.get_labels(filename)
                        for filename in os.listdir(self.filedir)]

    def __getitem__(self, index):
        mfcc, target = self.data[index], self.targets[index]

        if self.transform is not None:
            mfcc = self.transform(mfcc)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return mfcc, target

    def __len__(self):
        return len(self.data)

    def get_data(self, filename):
        filepath = os.path.join(self.filedir, filename)
        mfcc = get_mfcc(filepath, n_mfcc=26)

        return mfcc

    def get_labels(self, filename):
        tokens = filename.split(sep='_')
        # "This criterion(CrossEntropyLoss) expects a class index in the range [0, C-1] as the target for each value of a 1D tensor of size minibatch; if ignore_index is specified, this criterion also accepts this class index (this index may not necessarily be in the class range)". --from pytorch documentation.
        emotion_dict = {'angry': 0, 'fear': 1, 'happy': 2,
                        'neutral': 3, 'sad': 4, 'surprise': 5}

        if tokens[0] == "neutral":
            target = emotion_dict.get("neutral")
        elif tokens[0] == "emotional":
            target = emotion_dict.get(tokens[1])
        else:
            raise ValueError("请检查文件名！")
