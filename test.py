# from pathlib import Path

# import librosa
# import numpy as np
# import os
# import sys
# import torchaudio
# import matplotlib.pyplot as plt

# sys.path.extend(['opt/anaconda/etc/fonts/fonts.conf',
#                  'opt/anaconda/etc/fonts'])

# data_dir = "/run/media/dallasautumn/data/duan-qiu-yang/2019-2020-现代程序设计技术-大作业-数据/A类问题/A3-闻声知乐/CASIA情感语料库/CASIA情感语料库"
# print(os.listdir(data_dir))

# filename = '001.wav'
# wave_form, sample_rate = torchaudio.load(filename)
# print("Shape of waveform: {}".format(wave_form.size()))
# print("Sample rate of waveform: {}".format(sample_rate))

# # plt.ion()
# plt.figure()
# plt.plot(wave_form.t().numpy())

# specgram = torchaudio.transforms.Spectrogram()(wave_form)

# print("Shape of spectrogram: {}".format(specgram.size()))

# plt.figure()
# plt.imshow(specgram.log2()[0, :, :].numpy(), cmap='gray')
# plt.show()
import torch
import torch.nn.functional as F
a = torch.Tensor([0, 1, 2, 3, 4, 5])
print(F.one_hot(a, num_classes=6))
