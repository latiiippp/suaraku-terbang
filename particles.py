# filepath: d:\KULIAH\SEMESTER 6\Multimedia\tubes\suaraku-terbang\particles.py
"""
File Sistem Partikel.
Berisi kelas Particle untuk merepresentasikan satu partikel,
dan fungsi-fungsi untuk menambah, memperbarui, dan menggambar partikel.
"""
import numpy as np
import cv2
import config # Mengimpor list partikel global dari config.py

class Particle:
    """Kelas untuk merepresentasikan satu partikel dengan posisi, kecepatan, warna, ukuran, dan masa hidup."""
    def __init__(self, x, y, vx, vy, color, size, life):
        self.x = x          # Posisi x
        self.y = y          # Posisi y
        self.vx = vx        # Kecepatan horizontal
        self.vy = vy        # Kecepatan vertikal
        self.color = color  # Warna partikel (BGR tuple)
        self.size = size    # Ukuran awal partikel
        self.life = life    # Masa hidup partikel (dalam frame)
        self.max_life = life # Masa hidup maksimum untuk perhitungan alpha

    def update(self):
        """Memperbarui posisi dan masa hidup partikel."""
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Efek gravitasi sederhana
        self.life -= 1  # Kurangi masa hidup

    def is_alive(self):
        """Mengembalikan True jika partikel masih hidup, False jika tidak."""
        return self.life > 0
        
    def draw(self, img):
        """Menggambar partikel pada gambar 'img' jika masih hidup."""
        if self.is_alive():
            # Ukuran dan alpha partikel berkurang seiring berkurangnya masa hidup
            alpha = self.life / self.max_life 
            current_size = int(self.size * alpha) # Ukuran mengecil
            
            if current_size > 0:
                # Untuk menggambar dengan transparansi alpha, perlu blending manual jika tidak menggunakan RGBA image
                # Di sini, kita hanya menggambar lingkaran solid dengan ukuran yang disesuaikan
                # Jika 'img' adalah RGBA, blending bisa lebih canggih
                cv2.circle(img, (int(self.x), int(self.y)), current_size, self.color, -1)

def add_particles(x, y, color, count=10):
    """Menambahkan sejumlah 'count' partikel baru ke list global config.particles."""
    for _ in range(count):
        # Kecepatan acak
        vx = np.random.uniform(-3, 3) 
        vy = np.random.uniform(-5, -1) # Awalnya bergerak ke atas
        # Ukuran dan masa hidup acak
        size = np.random.randint(2, 6)
        life = np.random.randint(20, 40) # Masa hidup dalam frame
        config.particles.append(Particle(x, y, vx, vy, color, size, life))

def update_particles():
    """Memperbarui semua partikel dalam list global dan menghapus yang sudah mati."""
    # Buat list baru yang hanya berisi partikel yang masih hidup
    config.particles = [p for p in config.particles if p.is_alive()]
    # Update setiap partikel yang masih hidup
    for particle in config.particles:
        particle.update()

def draw_particles(img):
    """Menggambar semua partikel aktif pada gambar 'img'."""
    for particle in config.particles:
        particle.draw(img)