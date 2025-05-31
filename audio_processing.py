# filepath: d:\KULIAH\SEMESTER 6\Multimedia\tubes\suaraku-terbang\audio_processing.py
"""
File Pemrosesan Audio.
Berisi fungsi-fungsi untuk merekam suara, menganalisis frekuensi,
dan menentukan arah gerakan berdasarkan input suara.
"""
import sounddevice as sd
import numpy as np
import time
import config # Mengimpor variabel global dari config.py

def detect_sound_direction(duration=0.1, sample_rate=44100):
    """
    Merekam audio untuk durasi tertentu, menganalisis FFT untuk mendapatkan
    frekuensi dominan, energi bass/treble, dan menentukan arah (up/down/neutral).
    Memperbarui config.sound_info dan config.last_movement_direction.
    """
    try:
        # Merekam audio dari perangkat input default (atau device=31 jika spesifik)
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=1) # Sesuaikan device jika perlu
        sd.wait()  # Tunggu hingga rekaman selesai
        audio_data = recording.flatten() # Ubah array 2D menjadi 1D

        # Fast Fourier Transform (FFT) untuk analisis frekuensi
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft), 1/sample_rate) # Hitung frekuensi yang sesuai
        magnitudes = np.abs(fft) # Ambil magnitudo (kekuatan) dari setiap frekuensi

        # Ambil hanya frekuensi positif (simetris)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitudes = magnitudes[:len(magnitudes)//2]

        current_determined_direction = "neutral" 

        # Jika tidak ada suara terdeteksi atau magnitudo sangat kecil
        if len(positive_magnitudes) == 0 or np.sum(positive_magnitudes) < 1e-6: # Tambahkan threshold kecil
            config.sound_info["bass_energy"] = 0
            config.sound_info["treble_energy"] = 0
            config.sound_info["dominant_freq"] = 0
        else:
            # Cari frekuensi dengan magnitudo terbesar (frekuensi dominan)
            max_magnitude_idx = np.argmax(positive_magnitudes)
            dominant_freq = positive_freqs[max_magnitude_idx]
            
            # Hitung energi pada rentang frekuensi bass dan treble
            # Rentang frekuensi bass: 1 Hz - 150 Hz
            # Rentang frekuensi treble: > 150 Hz
            config.sound_info["bass_energy"] = np.sum(positive_magnitudes[(positive_freqs >= 1) & (positive_freqs <= 150)])
            config.sound_info["treble_energy"] = np.sum(positive_magnitudes[(positive_freqs > 150)])
            config.sound_info["dominant_freq"] = dominant_freq

            # Tentukan arah berdasarkan frekuensi dominan
            if dominant_freq == 0: # Jika tidak ada frekuensi dominan yang jelas
                current_determined_direction = "neutral"
            elif 1 <= dominant_freq <= 150: # Frekuensi rendah (bass)
                current_determined_direction = "down"
            elif dominant_freq > 150: # Frekuensi tinggi (treble)
                current_determined_direction = "up"
                
        config.last_movement_direction = current_determined_direction # Simpan arah terakhir
        return current_determined_direction
    except Exception as e:
        print(f"Audio error: {e}")
        # Jika terjadi error, kembalikan neutral dan set info suara ke nol
        config.sound_info["bass_energy"] = 0
        config.sound_info["treble_energy"] = 0
        config.sound_info["dominant_freq"] = 0
        config.last_movement_direction = "neutral"
        return "neutral"

def sound_thread():
    """
    Thread yang berjalan secara kontinyu untuk mendeteksi arah suara
    dan memperbarui variabel global config.sound_direction.
    """
    while True:
        # Panggil fungsi deteksi suara dan update variabel global
        config.sound_direction = detect_sound_direction()
        time.sleep(0.01) # Jeda singkat agar tidak membebani CPU