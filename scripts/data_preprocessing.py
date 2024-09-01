import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

def audio_to_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    plt.imsave(output_path, log_S, cmap='inferno')

def process_audio_files(input_dir, output_dir):
    for mod_type in ['AM', 'FM', 'QPSK']:
        mod_dir = os.path.join(input_dir, mod_type)
        output_mod_dir = os.path.join(output_dir, mod_type)
        if not os.path.exists(output_mod_dir):
            os.makedirs(output_mod_dir)
        
        for filename in os.listdir(mod_dir):
            if filename.endswith('.wav'):
                audio_path = os.path.join(mod_dir, filename)
                output_path = os.path.join(output_mod_dir, filename.replace('.wav', '.png'))
                audio_to_spectrogram(audio_path, output_path)

input_directory = 'data'
output_directory = 'spectrograms'
process_audio_files(input_directory, output_directory)
