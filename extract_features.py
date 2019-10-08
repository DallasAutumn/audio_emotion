import wave

import librosa
import pyaudio
import pyAudioAnalysis
import pydub
import soundfile
import torchaudio


def get_mfcc(audio_file_path, n_mfcc=26):
    """
    语音识别应用的标准做法是在20Hz-20kHz之间应用26个频率箱，且仅使用前13个进行分类。
    """
    y, sr = librosa.load(audio_file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc[:n_mfcc//2]
