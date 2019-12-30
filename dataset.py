import json
import logging
import os
import pickle
import random
import warnings
from os.path import join as path_join
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset

from extract_features import *

warnings.filterwarnings("ignore")

emotion_dict = json.load(open("class_index.json"), encoding='utf-8')


class AudioDataset(Dataset):
    """
    定义音频情感数据集
    从testandtrain中抽取
    """

    def __init__(self, root=None, train=None, transform=None, target_transform=None, n_jobs=1):
        """
        Args:
            root: str, the file directory.
            train: bool, specify training set or testing set.
            transform: a callable transform class, if not none, apply the transform to tensor.
            target_transform: same as transform, apply it to the target(label) instead.
            n_jobs: the number of workers to use when loading data, if not specified, use only 1 process as default.
        """

        self.root = root
        self.train = train  # "train" or "test"
        self.transform = transform
        self.target_transform = target_transform
        self.n_jobs = n_jobs

        if self.train is True:
            self.filedir = os.path.join(self.root, "train")
        elif self.train is False:
            self.filedir = os.path.join(self.root, "test")
        elif self.train is None:
            self.filedir = self.root
        else:
            raise ValueError("Expected a boolean value, got", type(self.train))

        self.data = Parallel(n_jobs=self.n_jobs, prefer='threads')(delayed(self.get_data)(filename)
                                                                   for filename in os.listdir(self.filedir))
        self.targets = Parallel(n_jobs=self.n_jobs, prefer='threads')(delayed(self.get_labels)(filename)
                                                                      for filename in os.listdir(self.filedir))

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Audio dataset with length {len(self.data)}"

    def get_data(self):
        """This method must be overridden by its subclasses."""

        raise NotImplementedError

    def get_labels(self, filename):
        tokens = filename.split(sep='_')
        # "This criterion(CrossEntropyLoss) expects a class index in the range [0, C-1] as the target for each value of a 1D tensor of size minibatch; if ignore_index is specified, this criterion also accepts this class index (this index may not necessarily be in the class range)". --from pytorch documentation.

        # emotion_dict = {'angry': 0, 'fear': 1, 'happy': 2,
        #                 'neutral': 3, 'sad': 4, 'surprise': 5}

        if tokens[0] == "neutral":
            target = int(emotion_dict.get("neutral"))
        elif tokens[0] == "emotional":
            target = int(emotion_dict.get(tokens[1]))
        else:
            raise ValueError("请检查文件名！")

        return target


class MfccDataset(AudioDataset):
    """提取mfcc特征"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"A mfcc dataset with length {len(self)}"

    def get_data(self, filename):
        filepath = os.path.join(self.filedir, filename)
        mfcc = get_mfcc(filepath, n_mfcc=26)

        return mfcc


class MelspecDataset(AudioDataset):
    """提取频谱图"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"A melspectrogram dataset with length {len(self)}"

    def get_data(self, filename):
        filepath = os.path.join(self.filedir, filename)
        specgram = get_spectrogram(filepath)

        return specgram


if __name__ == "__main__":
    from config import ROOT_DIR
    n_jobs = -1
    # start_time = time()
    # dataset = MfccDataset()
    # with open('./pickles/train_set.pkl', 'wb') as f:
    #     pickle.dump({'data': dataset.data, 'labels': dataset.targets}, f)
    # print("Total time : %.3f" % (time()-start_time))
    start = time()

    train_set = MfccDataset(
        root=path_join(ROOT_DIR, "相同文本300"), train=True, transform=None, n_jobs=n_jobs)

    val_set = MfccDataset(
        root=path_join(ROOT_DIR, "不同文本100"), transform=None, n_jobs=n_jobs)

    test_set = MfccDataset(
        root=path_join(ROOT_DIR, "相同文本300"), train=False, transform=None, n_jobs=n_jobs)

    print('total time:', time()-start)

    print(train_set)
    print(val_set)
    print(test_set)
