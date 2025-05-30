"""
File Utilitas Menggambar.
Berisi fungsi-fungsi dasar untuk operasi menggambar umum seperti
overlay transparan, menggambar persegi panjang bulat, panel gradien, dll.
"""
import cv2
import numpy as np
from config import color_schemes # Digunakan oleh draw_glassmorphism_panel

def overlay_transparent(background, overlay, x, y):
    """
    Menempelkan gambar 'overlay' (dengan alpha channel) ke gambar 'background'
    pada posisi (x,y) dengan benar menangani transparansi.
    """
    x = int(x)
    y = int(y)
    
    bh, bw = background.shape[:2]
    h_overlay, w_overlay = overlay.shape[:2]
    
    overlay_crop_x_start = 0
    overlay_crop_y_start = 0
    w_eff = w_overlay
    h_eff = h_overlay

    # Menangani jika overlay dimulai di luar batas kiri atau atas background
    if x < 0: 
        overlay_crop_x_start = -x 
        w_eff += x 
        x = 0
    if y < 0: 
        overlay_crop_y_start = -y 
        h_eff += y 
        y = 0

    if w_eff <= 0 or h_eff <= 0: # Jika lebar atau tinggi efektif menjadi nol atau negatif
        return background
    
    # Menangani jika overlay melewati batas kanan atau bawah background
    if x + w_eff > bw:
        w_eff = bw - x
    if y + h_eff > bh:
        h_eff = bh - y
        
    if w_eff <= 0 or h_eff <= 0: # Cek ulang setelah penyesuaian batas
        return background

    # Crop bagian overlay yang akan ditampilkan
    overlay_to_blend = overlay [
        int(overlay_crop_y_start) : int(overlay_crop_y_start + h_eff),
        int(overlay_crop_x_start) : int(overlay_crop_x_start + w_eff)
    ]
    
    # Jika hasil crop kosong atau tidak memiliki alpha channel
    if overlay_to_blend.size == 0 or overlay_to_blend.shape[2] < 4:
        # print(f"Warning: Overlay crop is empty or not RGBA. Shape: {overlay_to_blend.shape}")
        return background
    
    # Tentukan region di background yang akan ditimpa
    bg_y_start = int(y)
    bg_y_end = int(y + h_eff)
    bg_x_start = int(x)
    bg_x_end = int(x + w_eff)
    
    bg_region = background[bg_y_start:bg_y_end, bg_x_start:bg_x_end]
    
    # Pastikan dimensi bg_region dan overlay_to_blend sama
    if bg_region.shape[0] != overlay_to_blend.shape[0] or bg_region.shape[1] != overlay_to_blend.shape[1]:
        if bg_region.shape[0] > 0 and bg_region.shape[1] > 0: # Hanya resize jika bg_region valid
            overlay_to_blend = cv2.resize(overlay_to_blend, (bg_region.shape[1], bg_region.shape[0]))
        else:
            return background # Tidak bisa melakukan blending jika bg_region tidak valid

    # Proses blending
    alpha_overlay = overlay_to_blend[:, :, 3:] / 255.0 # Normalisasi alpha channel overlay
    color_overlay_pixels = overlay_to_blend[:, :, :3]  # Ambil channel warna (RGB) overlay
    
    bg_region_color = bg_region
    if bg_region.shape[2] == 4: # Jika background juga punya alpha, ambil RGB-nya saja
        bg_region_color = bg_region[:, :, :3]

    # Rumus blending alpha
    blended_color = color_overlay_pixels * alpha_overlay + bg_region_color * (1.0 - alpha_overlay)
    
    background[bg_y_start:bg_y_end, bg_x_start:bg_x_end, :3] = blended_color.astype(np.uint8)

    # Jika background memiliki alpha channel, update juga alpha channel-nya (opsional, tergantung kebutuhan)
    if background.shape[2] == 4:
        alpha_bg = bg_region[:, :, 3:] / 255.0 if bg_region.shape[2] == 4 else np.zeros_like(alpha_overlay)
        new_alpha_bg = alpha_overlay + alpha_bg * (1.0 - alpha_overlay)
        background[bg_y_start:bg_y_end, bg_x_start:bg_x_end, 3] = (new_alpha_bg * 255).astype(np.uint8)

    return background

def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=10, filled=False):
    """Menggambar persegi panjang dengan sudut tumpul."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Memastikan radius tidak terlalu besar untuk ukuran rectangle
    if radius > abs(x2-x1)/2: radius = abs(x2-x1)//2
    if radius > abs(y2-y1)/2: radius = abs(y2-y1)//2
    if radius < 0: radius = 0


    if filled:
        # Membuat mask untuk area yang diisi
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # Bagian tengah rectangle
        cv2.rectangle(mask, (x1+radius, y1), (x2-radius, y2), 255, -1)
        cv2.rectangle(mask, (x1, y1+radius), (x2, y2-radius), 255, -1)
        # Sudut-sudut tumpul (lingkaran)
        cv2.circle(mask, (x1+radius, y1+radius), radius, 255, -1)
        cv2.circle(mask, (x2-radius, y1+radius), radius, 255, -1)
        cv2.circle(mask, (x1+radius, y2-radius), radius, 255, -1)
        cv2.circle(mask, (x2-radius, y2-radius), radius, 255, -1)
        
        # Mengaplikasikan warna pada area yang di-mask
        # Perlu penanganan jika img adalah RGBA
        if img.shape[2] == 3:
            img[mask == 255] = color
        elif img.shape[2] == 4: # Jika RGBA, asumsikan color adalah (B,G,R) dan alpha 255
            img[mask == 255, :3] = color[:3]
            img[mask == 255, 3] = color[3] if len(color) == 4 else 255

    else: # Menggambar outline
        # Garis lurus
        cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness) # Atas
        cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness) # Bawah
        cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness) # Kiri
        cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness) # Kanan
        # Sudut (ellipse/arc)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness) # Kiri Atas
        cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness) # Kanan Atas
        cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)  # Kiri Bawah
        cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)   # Kanan Bawah

def draw_gradient_panel(img, pt1, pt2, color1, color2, alpha=0.8):
    """Menggambar panel dengan efek gradien warna dan transparansi."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Pastikan y2 > y1 agar np.linspace tidak error
    if y2 <= y1:
        # Jika tinggi panel 0 atau negatif, gambar rectangle solid sederhana jika alpha < 1
        if alpha < 1.0 and x2 > x1:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color1, -1)
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        elif x2 > x1: # Jika alpha 1.0, langsung gambar
             cv2.rectangle(img, (x1,y1), (x2,y2), color1, -1)
        return

    overlay = img.copy() # Buat salinan untuk blending transparan
    
    # Membuat gradien linear dari color1 ke color2
    gradient = np.linspace(0, 1, y2-y1).reshape(-1, 1) # Array dari 0 ke 1 sejumlah tinggi panel
    gradient_rgb = gradient * np.array(color2) + (1-gradient) * np.array(color1) # Interpolasi warna
    
    # Menggambar garis horizontal dengan warna gradien
    for i in range(y2-y1):
        color = tuple(map(int, gradient_rgb[i])) # Konversi warna ke tuple integer
        cv2.rectangle(overlay, (x1, y1+i), (x2, y1+i+1), color, -1) # Gambar garis setebal 1px
    
    # Mengaplikasikan transparansi
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_glassmorphism_panel(img, pt1, pt2, blur_strength=15, alpha=0.3):
    """Menciptakan efek panel glassmorphism (kaca buram)."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Memastikan koordinat valid dan dalam batas gambar
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1: # Jika region tidak valid
        return
    
    blur_strength = max(1, blur_strength) # Kekuatan blur minimal 1
    if blur_strength % 2 == 0: # Harus ganjil untuk GaussianBlur kernel
        blur_strength += 1
    
    bg_region = img[y1:y2, x1:x2].copy() # Ekstrak region latar belakang
    
    # Sesuaikan kekuatan blur jika region terlalu kecil
    if bg_region.shape[0] < blur_strength or bg_region.shape[1] < blur_strength:
        blur_strength = min(bg_region.shape[0], bg_region.shape[1])
        if blur_strength < 3: blur_strength = 1
        elif blur_strength % 2 == 0: blur_strength -=1 
        if blur_strength <= 0: return # Tidak bisa blur jika kernel 0 atau negatif

    if bg_region.shape[0] > 0 and bg_region.shape[1] > 0: # Hanya proses jika region valid
        blurred = cv2.GaussianBlur(bg_region, (blur_strength, blur_strength), 0) # Aplikasikan blur
        
        # Membuat overlay putih semi-transparan untuk efek kaca
        overlay_color = np.ones_like(blurred) * 40 # Base color (gelap)
        overlay_color[:, :, 0] = 60  # Sedikit tint biru-abu
        overlay_color[:, :, 1] = 60
        overlay_color[:, :, 2] = 80
        
        # Blend overlay dengan background yang sudah diblur
        glass_effect = cv2.addWeighted(blurred, 1-alpha, overlay_color, alpha, 0)
        
        img[y1:y2, x1:x2] = glass_effect # Timpa region di gambar utama
        
        # Tambahkan border tipis untuk memperjelas panel
        border_color_val = color_schemes["primary"] # Ambil dari config
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color_val, 2)
        
        # Tambahkan inner glow (efek cahaya di dalam border)
        inner_overlay = img.copy()
        cv2.rectangle(inner_overlay, (x1+2, y1+2), (x2-2, y2-2), border_color_val, 1) # Border lebih tipis di dalam
        cv2.addWeighted(img, 0.9, inner_overlay, 0.1, 0, img) # Blend dengan alpha rendah