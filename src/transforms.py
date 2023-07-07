import torchaudio.transforms as tt


def rawspec():
    return tt.Spectrogram(n_fft=1024, win_length=1024, hop_length=512, power=1.0)


def powerspec():
    return tt.Spectrogram(n_fft=1024, win_length=1024, hop_length=512, power=2.0)


def melspec():
    raise NotImplementedError


def mfcc():
    raise NotImplementedError


def lfcc():
    raise NotImplementedError


def imfcc():
    raise NotImplementedError
