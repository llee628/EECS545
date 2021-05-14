import numpy as np
import scipy.signal
from scipy.fftpack import dct


"""
The code is modify from
[1] https://github.com/ashishpatel26/Audio-Masking-Methods/blob/master/TFMethods.py
[2] https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""


def frft(f, a):
    """
    Calculate the fast fractional fourier transform.
    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power
    Returns
    -------
    data : numpy array
        The transformed signal.
    References
    ---------
     .. [1] This algorithm implements `frft.m` from
        https://nalag.cs.kuleuven.be/research/software/FRFT/
    """
    ret = np.zeros_like(f, dtype=np.complex)
    f = f.copy().astype(np.complex)
    N = len(f)
    shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
    sN = np.sqrt(N)
    a = np.remainder(a, 4.0)

    # Special cases
    if a == 0.0:
        return f
    if a == 2.0:
        return np.flipud(f)
    if a == 1.0:
        ret[shft] = np.fft.fft(f[shft]) / sN
        return ret
    if a == 3.0:
        ret[shft] = np.fft.ifft(f[shft]) * sN
        return ret

    # reduce to interval 0.5 < a < 1.5
    if a > 2.0:
        a = a - 2.0
        f = np.flipud(f)
    if a > 1.5:
        a = a - 1
        f[shft] = np.fft.fft(f[shft]) / sN
    if a < 0.5:
        a = a + 1
        f[shft] = np.fft.ifft(f[shft]) * sN

    # the general case for 0.5 < a < 1.5
    alpha = a * np.pi / 2
    tana2 = np.tan(alpha / 2)
    sina = np.sin(alpha)
    f = np.hstack((np.zeros(N - 1), sincinterp(f), np.zeros(N - 1))).T

    # chirp premultiplication
    chrp = np.exp(-1j * np.pi / N * tana2 / 4 *
                     np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
    f = chrp * f

    # chirp convolution
    c = np.pi / N / sina / 4
    ret = scipy.signal.fftconvolve(
        np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
        f
    )
    ret = ret[4 * N - 4:8 * N - 7] * np.sqrt(c / np.pi)

    # chirp post multiplication
    ret = chrp * ret

    # normalizing constant
    ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]

    return ret


def sincinterp(x):
    N = len(x)
    y = np.zeros(2 * N - 1, dtype=x.dtype)
    y[:2 * N:2] = x
    xint = scipy.signal.fftconvolve(
        y[:2 * N],
        np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
    )
    return xint[2 * N - 3: -2 * N + 3]



def mfcc_fft(signal, a=0.97): # a = 0.9 - 0.99
    sample_rate = 16000

    # Pre-Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_size = 0.025
    frame_stride = 0.01

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Window
    frames *= np.hamming(frame_length)

    # FFT
    NFFT = 512
    # mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    # pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    # pow_frames = np.zeros(frames.shape)


    mag_frames_temp = np.zeros(frames.shape)
    # print(frames.shape)
    # pow_frames_temp = Disfrft(frames, 0.97)

    for kk in range(frames.shape[0]):
        mag_frames_temp[kk, :] = frft(frames[kk, :], a)
        # mag_frames_temp[kk, :] = Disfrft(frames[kk, :], 0.97, NFFT)
    mag_frames = mag_frames_temp[:, 0:257]
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # pow_frames = pow_frames_temp


    # Filter Banks
    nfilt = 26
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    # filter_banks = 20 * np.log10(filter_banks)  # dB
    filter_banks = np.log(filter_banks)

    # DCT
    num_ceps = 13
    cep_lifter = 22
    # mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]

    nframes, ncoeff = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*


    # mean normalization
    # filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    # mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc
