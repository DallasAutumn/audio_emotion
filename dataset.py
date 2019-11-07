import os
import pickle
import random
import warnings
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from extract_features import get_mfcc
from transforms import ToTensor

warnings.filterwarnings("ignore")


class AudioDataset(Dataset):
    """
    定义音频情感数据集
    从testandtrain中抽取
    Thanks for the previous work of Weng Di :)
    """

    def __init__(self, root=None, train=None, transform=None, target_transform=None):
        self.root = root
        self.train = train  # "train" or "test"
        self.transform = transform
        self.target_transform = target_transform

        if self.train is True:
            self.filedir = os.path.join(self.root, "train")
        elif self.train is False:
            self.filedir = os.path.join(self.root, "test")
        elif self.train is None:
            self.filedir = self.root
        else:
            raise ValueError("Expected a boolean value, got", type(self.train))

        self.data = [self.get_data(filename)
                     for filename in os.listdir(self.filedir)]
        self.targets = [self.get_labels(filename)
                        for filename in os.listdir(self.filedir)]

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

    def get_data(self):
        raise NotImplementedError

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

        return target


class MfccDataset(AudioDataset):
    """提取mfcc特征"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self, filename):
        filepath = os.path.join(self.filedir, filename)
        mfcc = get_mfcc(filepath, n_mfcc=26)

        return mfcc


if __name__ == "__main__":
    start_time = time()
    dataset = MfccDataset()
    with open('./pickles/train_set.pkl', 'wb') as f:
        pickle.dump({'data': dataset.data, 'labels': dataset.targets}, f)
    print("Total time : %.3f" % (time()-start_time))
