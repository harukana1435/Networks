import librosa
import numpy as np
import torch


def generate_spectrogram_torch(y):
    fft_size = 1024
    hop_size = 256
    win_size=1024
    y = torch.FloatTensor(y)
    
    # 入力の音声が-1〜1に収まっていない場合に警告
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
        
    hann_window = torch.hann_window(win_size)

    # STFTを計算する
    spec = torch.stft(y, fft_size, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    
    # 複素数の絶対値を計算する
    mag = torch.abs(spec)
    mag = convert_to_db(mag)
    mag = mag[:, :fft_size//2, :]
    
    return mag

def generate_spectrum_torch(y):
    fft_size = 1024
    
    y = torch.FloatTensor(y)
    
    # 入力の音声が-1〜1に収まっていない場合に警告
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
        
    spec = torch.fft.fft(y, n=fft_size)
    
    # 複素数の絶対値を計算する
    mag = torch.abs(spec)
    mag = convert_to_db(mag)
    mag = mag[:, :fft_size//2]
    
    return mag

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples

def convert_to_db(mag, eps=1e-10):
    return 20 * np.log10(np.maximum(mag, eps))

def generate_spectrogram(audio):
    complex_spectrogram = librosa.core.stft(audio, n_fft=1024, hop_length=256, win_length=1024, center=True)
    mag = np.abs(complex_spectrogram)
    mag = convert_to_db(mag)
    phase = np.angle(complex_spectrogram)
    spectro_two_channel = np.concatenate((mag, phase), axis=0)
    return spectro_two_channel

def load_audio(audio_path, length = 32684, sample_rate=48000):
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
    clip_audio = audio[:, :length]
    left_audio, right_audio = clip_audio[0, :], clip_audio[1, :]
    left_audio = np.expand_dims(left_audio, axis=0)
    right_audio = np.expand_dims(right_audio, axis=0)
    return left_audio, right_audio

if __name__=="__main__":
    left_audio, right_audio = load_audio(r"C:\Users\okano\intern\Networks\data\intern_binaural.wav")
    left_audio = normalize(left_audio)
    right_audio = normalize(right_audio)
    left_spectrogram = generate_spectrogram_torch(left_audio)
    right_spectrogram = generate_spectrogram_torch(right_audio)
    input_spectrogram = torch.concatenate((left_spectrogram, right_spectrogram), axis=0)
    print(input_spectrogram.shape)
    
    left_spectrum = generate_spectrum_torch(left_audio)
    right_spectrum = generate_spectrum_torch(right_audio)
    gt_spectrum = torch.concatenate((left_spectrum, right_spectrum), axis=0)
    gt_hrtf_spectrogram = gt_spectrum.unsqueeze(-1).repeat(1, 1, 128)
    print(gt_hrtf_spectrogram.shape)
    
    
    
    