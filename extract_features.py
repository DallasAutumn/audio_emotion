import wave

import librosa
import matplotlib.pyplot as plt
import pyaudio
import pyAudioAnalysis
import pydub
import soundfile
import torchaudio


def get_mfcc(audio_file_path, n_mfcc=26):
    """
    提取梅尔倒频谱
    语音识别应用的标准做法是在20Hz-20kHz之间应用26个频率箱，且仅使用前13个进行分类。
    """
    y, sr = librosa.load(audio_file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc[:n_mfcc//2]


def get_spectrogram(audio_file_path):
    """提取梅尔特征频谱图"""
    waveform, sample_rate = torchaudio.load(audio_file_path)
    specgram = torchaudio.transforms.Spectrogram()(waveform)
    return specgram


if __name__ == "__main__":
    sg1 = get_spectrogram('001.wav')
    sg2 = get_spectrogram('201.wav')
    print(sg1.shape, sg2.shape)
