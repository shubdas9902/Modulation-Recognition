import numpy as np
from scipy.io.wavfile import write
import os

# Parameters
sampling_rate = 22050  # Sampling rate in Hz
duration = 2.0         # Duration in seconds

# Directory to save generated samples
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_am_signal(frequency, mod_index, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    carrier = np.sin(2 * np.pi * frequency * t)
    modulator = 1 + mod_index * np.sin(2 * np.pi * 1 * t)  # 1 Hz modulator frequency
    signal = modulator * carrier
    return signal

def generate_fm_signal(frequency, mod_index, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    modulator = mod_index * np.sin(2 * np.pi * 1 * t)  # 1 Hz modulator frequency
    signal = np.sin(2 * np.pi * frequency * t + modulator)
    return signal

def generate_qpsk_signal(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    num_symbols = int(sample_rate * duration)
    symbol_duration = duration / num_symbols
    symbols = np.zeros(num_symbols)
    for i in range(num_symbols):
        phase = (i % 4) * (np.pi / 2)  # Phase shifts of 0, π/2, π, 3π/2
        symbols[i] = np.sin(2 * np.pi * frequency * t[i] + phase)
    signal = np.interp(t, np.linspace(0, duration, num_symbols), symbols)
    return signal

def save_signal(signal, filename, sample_rate):
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    write(filename, sample_rate, signal)

def generate_varied_signals():
    modulation_types = ['AM', 'FM', 'QPSK']
    frequencies = [500, 1000, 1500]  # Different frequencies for variety
    mod_indices = [0.5, 1.0, 2.0]    # Different modulation indices for AM and FM

    for mod_type in modulation_types:
        mod_dir = os.path.join(output_dir, mod_type)
        if not os.path.exists(mod_dir):
            os.makedirs(mod_dir)
        
        for freq in frequencies:
            for index in range(10):  # Generate 10 samples for each frequency
                filename = f'{mod_type}_{freq}_{index}.wav'
                file_path = os.path.join(mod_dir, filename)
                
                if mod_type == 'AM':
                    mod_index = np.random.choice(mod_indices)
                    signal = generate_am_signal(frequency=freq, mod_index=mod_index, duration=duration, sample_rate=sampling_rate)
                elif mod_type == 'FM':
                    mod_index = np.random.choice(mod_indices)
                    signal = generate_fm_signal(frequency=freq, mod_index=mod_index, duration=duration, sample_rate=sampling_rate)
                elif mod_type == 'QPSK':
                    signal = generate_qpsk_signal(frequency=freq, duration=duration, sample_rate=sampling_rate)
                
                save_signal(signal, file_path, sampling_rate)

if __name__ == "__main__":
    generate_varied_signals()
    print("Varied signals have been generated and saved.")
