import numpy as np
import glob
import librosa


def read_wave_file(filepath):
    return librosa.load(filepath)


def same_class_augmentation(wave, class_dir):
    """
    同类增强
    Perform same class augmentation of the wave by loading a random segment
    from the class_dir and additively combine the wave with that segment.
    """
    sig_paths = glob.glob(os.path.join(class_dir, "*.wav"))
    aug_sig_path = np.random.choice(sig_paths, 1, replace=False)[0]
    (fs, aug_sig) = read_wave_file(aug_sig_path)
    alpha = np.random.rand()
    wave = (1.0-alpha)*wave + alpha*aug_sig
    return wave


def noise_augmentation(wave, noise_dir):
    """
    噪声增强
    Perform noise augmentation of the wave by loading three noise segments
    from the noise_dir and add these on top of the wave with a dampening factor
    of 0.4
    """
    noise_paths = glob.glob(os.path.join(noise_dir, "*.wav"))
    aug_noise_paths = np.random.choice(noise_paths, 3, replace=False)
    dampening_factor = 0.4
    for aug_noise_path in aug_noise_paths:
        (fs, aug_noise) = read_wave_file(aug_noise_path)
        wave = wave + aug_noise*dampening_factor
    return wave


def time_shift_spectrogram(spectrogram):
    """ 
    频谱图时移增强
    Shift a spectrogram along the time axis in the spectral-domain at random
    """
    nb_cols = spectrogram.shape[1]
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, nb_shifts, axis=1)


def pitch_shift_spectrogram(spectrogram):
    """ 
    频谱图音高变换增强
    Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = spectrogram.shape[0]
    max_shifts = nb_cols//20  # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(spectrogram, nb_shifts, axis=0)
