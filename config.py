# filepath: d:\KULIAH\SEMESTER 6\Multimedia\tubes\suaraku-terbang\config.py
"""
File Konfigurasi dan State Global Game.
Berisi semua variabel global, pengaturan warna, level, skor,
dan state game lainnya yang perlu diakses dari berbagai modul.
"""

# State Audio
sound_direction = "neutral"  # Arah suara yang terdeteksi (up, down, neutral)
last_movement_direction = "neutral" # Arah gerakan terakhir berdasarkan suara
sound_info = {"bass_energy": 0, "treble_energy": 0, "dominant_freq": 0} # Informasi detail dari audio

# Pengaturan Level
current_level = 1  # Level game saat ini
max_level = 5      # Jumlah maksimum level

# Sistem Skor
current_score = 0      # Skor total yang dikumpulkan dari level sebelumnya
score_this_level = 0   # Skor yang sedang dikumpulkan di level saat ini (maks 100 per level)

# State Game
game_over = False      # Status apakah game telah berakhir
game_started = False   # Status apakah game telah dimulai (melewati start screen)

# Variabel Animasi UI
collision_flash = 0    # Timer untuk efek kilat saat tabrakan
level_up_flash = 0     # Timer untuk efek kilat saat naik level
ui_animations = {"score_pulse": 0, "level_glow": 0, "background_wave": 0, "particle_time": 0} # State untuk animasi UI

# Sistem Partikel dan Jejak Bola
particles = []         # List untuk menyimpan objek partikel aktif
ball_trail = []        # List untuk menyimpan posisi jejak bola
max_trail_length = 15  # Panjang maksimum jejak bola

# Skema Warna
# Digunakan untuk konsistensi tampilan elemen-elemen game
color_schemes = {
    "primary": (100, 150, 255),       # Warna utama (misalnya, untuk border, aksen)
    "secondary": (50, 200, 150),      # Warna sekunder
    "accent": (255, 100, 150),        # Warna aksen (misalnya, untuk jejak bola)
    "background_dark": (15, 15, 25),  # Warna latar belakang gelap
    "background_light": (25, 25, 35), # Warna latar belakang terang
    "text_primary": (255, 255, 255),  # Warna teks utama (putih)
    "text_secondary": (180, 180, 200),# Warna teks sekunder (abu-abu muda)
    "success": (100, 255, 100),       # Warna untuk indikasi sukses (hijau)
    "warning": (255, 200, 100),       # Warna untuk peringatan (kuning/oranye)
    "danger": (255, 100, 100)         # Warna untuk bahaya atau error (merah)
}

# Pengaturan Penghalang per Level
# Menentukan posisi y atas dan bawah penghalang relatif terhadap tinggi layar
# Format: {level: (faktor_posisi_atas, faktor_posisi_bawah)}
level_barrier_settings = {
    1: (0.30, 0.70),  # Level 1: Celah 40% dari tinggi layar
    2: (0.35, 0.65),  # Level 2: Celah 30%
    3: (0.40, 0.60),  # Level 3: Celah 20%
    4: (0.425, 0.575), # Level 4: Celah 15%
    5: (0.45, 0.55)   # Level 5: Celah 10%
}