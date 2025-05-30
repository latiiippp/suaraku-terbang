"""
File Fungsi Menggambar Elemen Game dan UI.
Berisi semua fungsi yang bertanggung jawab untuk menggambar berbagai
komponen visual game seperti HUD, panel informasi, layar game over, dll.
"""
import cv2
import numpy as np
import time # Digunakan oleh draw_sound_visualizer dan draw_start_screen (untuk animasi)
import config # Akses ke variabel global seperti color_schemes, ui_animations, dll.
import visuals_utils # Utilitas menggambar dasar
import particles # Untuk menambah partikel saat event tertentu (misalnya, di draw_split_interface)

def draw_modern_hud(img, width, height, fps, score, level, sound_info, sound_direction, collision_flash, level_up_flash):
    """Menggambar HUD modern dengan panel skor, level, dan info audio (versi lama, tidak dipakai jika split interface aktif)."""
    # Update animasi internal HUD (jika ada)
    config.ui_animations["score_pulse"] = (config.ui_animations["score_pulse"] + 1) % 60
    config.ui_animations["level_glow"] = (config.ui_animations["level_glow"] + 1) % 120
    
    # Panel Atas - Skor dan Level
    panel_height = 80
    visuals_utils.draw_gradient_panel(img, (0, 0), (width, panel_height), (30, 30, 40), (20, 20, 30), 0.85)
    
    # Skor dengan efek denyut
    score_pulse = abs(np.sin(config.ui_animations["score_pulse"] * 0.1)) * 20
    score_color = (50 + int(score_pulse), 255, 50 + int(score_pulse))
    cv2.putText(img, "SCORE", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(img, f"{score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 2)
    
    # Level dengan efek glow
    level_glow = abs(np.sin(config.ui_animations["level_glow"] * 0.05)) * 30
    level_color = (50, 150 + int(level_glow), config.color_schemes["primary"][2]) # Ambil komponen biru dari primary
    cv2.putText(img, "LEVEL", (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(img, f"{level}/{config.max_level}", (width - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, level_color, 2)
    
    # Progress bar level
    bar_width_hud = 100 # Hindari konflik nama variabel
    bar_height_hud = 8
    bar_x_hud = width - 130
    bar_y_hud = 60
    visuals_utils.draw_rounded_rectangle(img, (bar_x_hud, bar_y_hud), (bar_x_hud + bar_width_hud, bar_y_hud + bar_height_hud), (60, 60, 60), filled=True, radius=4)
    progress_hud = level / config.max_level
    progress_width_hud = int(bar_width_hud * progress_hud)
    if progress_width_hud > 0:
        visuals_utils.draw_rounded_rectangle(img, (bar_x_hud, bar_y_hud), (bar_x_hud + progress_width_hud, bar_y_hud + bar_height_hud), level_color, filled=True, radius=4)
    
    # Panel Samping - Info Audio (Contoh, bisa disesuaikan)
    # ... (Implementasi panel audio jika diperlukan di HUD ini) ...

    # Efek kilat tabrakan dan naik level
    if collision_flash > 0:
        flash_alpha = collision_flash / 30.0
        flash_overlay = img.copy()
        cv2.rectangle(flash_overlay, (0, 0), (width, height), config.color_schemes["danger"], -1)
        cv2.addWeighted(img, 1-flash_alpha*0.3, flash_overlay, flash_alpha*0.3, 0, img)
    
    if level_up_flash > 0:
        flash_alpha = level_up_flash / 60.0
        flash_overlay = img.copy()
        cv2.rectangle(flash_overlay, (0, 0), (width, height), config.color_schemes["success"], -1)
        cv2.addWeighted(img, 1-flash_alpha*0.2, flash_overlay, flash_alpha*0.2, 0, img)


def draw_modern_barriers(img, width, height, top_y, bottom_y, thickness):
    """Menggambar penghalang (barriers) dengan gaya modern dan efek glow."""
    barrier_color = config.color_schemes["primary"]
    glow_color = tuple(c // 2 for c in barrier_color) # Warna glow lebih gelap
    
    # Efek glow untuk barrier
    for i in range(3): # Beberapa layer glow
        glow_thickness = thickness + (3-i) * 2 # Ketebalan glow berkurang
        alpha = 0.3 - i * 0.1 # Transparansi glow berkurang
        overlay = img.copy()
        # Gambar garis glow
        cv2.line(overlay, (0, top_y), (width, top_y), glow_color, glow_thickness)
        cv2.line(overlay, (0, bottom_y), (width, bottom_y), glow_color, glow_thickness)
        cv2.addWeighted(img, 1-alpha, overlay, alpha, 0, img) # Blend glow ke gambar utama
    
    # Gambar garis barrier utama
    cv2.line(img, (0, top_y), (width, top_y), barrier_color, thickness)
    cv2.line(img, (0, bottom_y), (width, bottom_y), barrier_color, thickness)

def draw_modern_game_over(width, height, final_score, max_score_achieved):
    """Membuat gambar untuk layar Game Over dengan skor akhir."""
    game_over_img = np.zeros((height, width, 3), dtype=np.uint8) # Latar belakang hitam
    
    # Latar belakang gradien
    visuals_utils.draw_gradient_panel(game_over_img, (0, 0), (width, height), (20, 20, 30), (40, 40, 60), 1.0)
    
    # Panel utama Game Over
    panel_width_go = 400 # Hindari konflik nama
    panel_height_go = 300
    panel_x_go = (width - panel_width_go) // 2
    panel_y_go = (height - panel_height_go) // 2
    
    visuals_utils.draw_gradient_panel(game_over_img, (panel_x_go, panel_y_go), (panel_x_go + panel_width_go, panel_y_go + panel_height_go), 
                               (60, 60, 80), (40, 40, 60), 0.95)
    visuals_utils.draw_rounded_rectangle(game_over_img, (panel_x_go, panel_y_go), (panel_x_go + panel_width_go, panel_y_go + panel_height_go), 
                                  config.color_schemes["text_secondary"], 3, 15)
    
    # Teks "GAME OVER"
    text_center_x_go = panel_x_go + panel_width_go // 2
    text_y_go = panel_y_go + 80
    
    # Efek glow untuk judul
    title_text_go = "GAME OVER"
    (w_text, h_text), _ = cv2.getTextSize(title_text_go, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    for i in range(3): # Beberapa layer glow
        glow_alpha = 0.4 - i*0.1
        overlay = game_over_img.copy()
        cv2.putText(overlay, title_text_go, (text_center_x_go - w_text//2, text_y_go), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, config.color_schemes["danger"], 5 + i*2) # Glow lebih tebal
        cv2.addWeighted(game_over_img, 1-glow_alpha, overlay, glow_alpha, 0, game_over_img)

    cv2.putText(game_over_img, title_text_go, (text_center_x_go - w_text//2, text_y_go), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, config.color_schemes["text_primary"], 3)
    
    # Tampilan Skor Akhir
    score_y_go = text_y_go + 80
    score_text = f"Final Score: {final_score}"
    (w_score_text, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(game_over_img, score_text, (text_center_x_go - w_score_text//2, score_y_go), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, config.color_schemes["success"], 2)
    
    # Instruksi keluar
    instruction_y_go = score_y_go + 60
    instr_text = "Press any key to exit"
    (w_instr_text, _), _ = cv2.getTextSize(instr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(game_over_img, instr_text, (text_center_x_go - w_instr_text//2, instruction_y_go), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.color_schemes["text_secondary"], 2)
    
    return game_over_img

def add_ball_trail(x, y):
    """Menambahkan posisi bola saat ini ke list jejak bola (config.ball_trail)."""
    config.ball_trail.append((x, y))
    # Jika jejak terlalu panjang, hapus elemen tertua
    if len(config.ball_trail) > config.max_trail_length:
        config.ball_trail.pop(0)

def draw_ball_trail(img):
    """Menggambar jejak bola berdasarkan posisi yang tersimpan di config.ball_trail."""
    if len(config.ball_trail) < 2: # Perlu minimal 2 titik untuk menggambar garis
        return
    
    for i in range(1, len(config.ball_trail)):
        # Alpha dan ketebalan berkurang untuk segmen jejak yang lebih tua
        alpha_trail = i / len(config.ball_trail) 
        thickness_trail = int(3 * alpha_trail) # Ketebalan maksimal 3px
        
        if thickness_trail > 0:
            # Warna jejak dengan alpha yang disesuaikan
            trail_color_base = config.color_schemes["accent"]
            # Membuat warna dengan alpha (jika gambar mendukung RGBA, ini bisa lebih baik)
            # Untuk BGR, kita bisa mencoba memudarkan warnanya
            current_trail_color = tuple(int(c * alpha_trail) for c in trail_color_base) 

            pt1 = config.ball_trail[i-1]
            pt2 = config.ball_trail[i]
            
            start_point = (int(pt1[0]), int(pt1[1]))
            end_point = (int(pt2[0]), int(pt2[1]))
            
            cv2.line(img, start_point, end_point, current_trail_color, thickness_trail)

def draw_animated_background(img, width, height):
    """Menggambar latar belakang animasi dengan pola gelombang."""
    config.ui_animations["background_wave"] = (config.ui_animations["background_wave"] + 1) % 360 # Update animasi gelombang
    
    wave_offset = config.ui_animations["background_wave"] * 0.02 # Offset untuk pergerakan gelombang
    
    # Menggambar lingkaran kecil untuk membentuk pola gelombang
    for y_coord in range(0, height, 20): # Iterasi per 20 piksel y
        for x_coord in range(0, width, 20): # Iterasi per 20 piksel x
            # Hitung nilai gelombang berdasarkan posisi dan offset waktu
            wave = np.sin((x_coord * 0.01) + wave_offset) * np.cos((y_coord * 0.01) + wave_offset)
            intensity = int(abs(wave) * 15) # Intensitas warna berdasarkan nilai gelombang
            
            # Warna dasar dari config, ditambah intensitas
            base_bg_color = config.color_schemes["background_dark"]
            current_bg_color = (
                min(255, base_bg_color[0] + intensity), # Pastikan tidak melebihi 255
                min(255, base_bg_color[1] + intensity),
                min(255, base_bg_color[2] + intensity)
            )
            cv2.circle(img, (x_coord, y_coord), 2, current_bg_color, -1) # Gambar lingkaran kecil

def draw_sound_visualizer(img, x, y, width, height, sound_info):
    """Menggambar visualisasi suara sederhana (misalnya, untuk panel audio)."""
    bar_count = 20 # Jumlah bar visualizer
    bar_width_viz = width // bar_count # Lebar setiap bar, hindari konflik nama
    
    # Rasio energi bass dan treble (normalisasi)
    bass_ratio = min(sound_info['bass_energy'] / 1000, 1.0) # Batasi maksimal 1.0
    treble_ratio = min(sound_info['treble_energy'] / 1000, 1.0)
    
    for i in range(bar_count):
        # Tinggi bar bervariasi berdasarkan frekuensi (simulasi) dan waktu
        if i < bar_count // 2: # Bar untuk bass
            bar_height_val = int(height * bass_ratio * (0.5 + 0.5 * np.sin(i * 0.5 + time.time() * 2)))
        else: # Bar untuk treble
            bar_height_val = int(height * treble_ratio * (0.5 + 0.5 * np.sin(i * 0.5 + time.time() * 3)))
        
        bar_x_viz = x + i * bar_width_viz # Posisi x bar, hindari konflik nama
        bar_y_viz = y + height - bar_height_val # Posisi y bar (dari bawah), hindari konflik nama
        
        # Warna gradien dari bass (biru/ungu) ke treble (merah/oranye)
        ratio_color = i / bar_count
        current_bar_color = (
            int(config.color_schemes["secondary"][0] * (1-ratio_color) + config.color_schemes["warning"][0] * ratio_color),
            int(config.color_schemes["secondary"][1] * (1-ratio_color) + config.color_schemes["warning"][1] * ratio_color),
            int(config.color_schemes["secondary"][2] * (1-ratio_color) + config.color_schemes["warning"][2] * ratio_color)
        )
        cv2.rectangle(img, (bar_x_viz, bar_y_viz), (bar_x_viz + bar_width_viz - 2, y + height), current_bar_color, -1)


def draw_frequency_display(img, x, y, width, height, sound_info):
    """Menggambar tampilan frekuensi dominan pada panel informasi."""
    freq = int(sound_info['dominant_freq'])
    
    # Latar belakang panel frekuensi
    panel_color_freq = (30, 30, 50) # Warna spesifik atau dari config
    cv2.rectangle(img, (x, y), (x + width, y + height), panel_color_freq, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), config.color_schemes["primary"], 2) # Border
    
    # Judul panel
    cv2.putText(img, "FREQUENCY", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.color_schemes["text_secondary"], 2)
    
    # Teks angka frekuensi besar
    freq_text = f"{freq}"
    # Warna teks frekuensi berdasarkan rentang (rendah/tinggi/netral)
    freq_text_color = config.color_schemes["warning"] if freq > 150 else config.color_schemes["secondary"] if freq > 0 else config.color_schemes["text_secondary"]
    
    font_scale_freq = 1.8 # Ukuran font angka frekuensi
    thickness_freq = 2    # Ketebalan font
    text_y_pos_freq = y + 75   # Posisi Y teks frekuensi

    # Efek glow untuk angka frekuensi
    for i in range(3): # Beberapa layer glow
        glow_alpha_freq = 0.3 - i * 0.1
        glow_offset_freq = i * 1
        overlay_freq = img.copy()
        cv2.putText(overlay_freq, freq_text, (x + 15 + glow_offset_freq, text_y_pos_freq + glow_offset_freq), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_freq, freq_text_color, thickness_freq + 1)
        cv2.addWeighted(img, 1-glow_alpha_freq, overlay_freq, glow_alpha_freq, 0, img)
    
    # Gambar teks frekuensi utama
    cv2.putText(img, freq_text, (x + 15, text_y_pos_freq), cv2.FONT_HERSHEY_SIMPLEX, font_scale_freq, config.color_schemes["text_primary"], thickness_freq)
    # Tambahkan "Hz" di sebelah angka
    (w_freq_text, _), _ = cv2.getTextSize(freq_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_freq, thickness_freq)
    cv2.putText(img, "Hz", (x + 15 + w_freq_text + 5, text_y_pos_freq), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.color_schemes["text_secondary"], 1)

def draw_game_info_panel(img, x, y, width, height, score, level, fps): # fps tidak lagi dipakai tapi argumen dipertahankan
    """Menggambar panel informasi game (skor, level)."""
    panel_color_info = (25, 25, 40) # Warna spesifik atau dari config
    cv2.rectangle(img, (x, y), (x + width, y + height), panel_color_info, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), config.color_schemes["primary"], 2) # Border
    
    cv2.putText(img, "GAME INFO", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.color_schemes["text_primary"], 2)
    
    current_y_info = y + 50 # Posisi Y saat ini untuk elemen panel
    
    # Bagian Skor
    cv2.putText(img, "SCORE:", (x + 10, current_y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.color_schemes["text_secondary"], 1)
    score_text_color = config.color_schemes["success"]
    cv2.putText(img, f"{score}", (x + 90, current_y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_text_color, 2)
    current_y_info += 35
    
    # Bagian Level
    cv2.putText(img, "LEVEL:", (x + 10, current_y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.color_schemes["text_secondary"], 1)
    level_text_color = config.color_schemes["primary"]
    cv2.putText(img, f"{level}/{config.max_level}", (x + 90, current_y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.8, level_text_color, 2)
    current_y_info += 35
    
    # Progress bar level
    bar_width_info = width - 20 # Lebar progress bar, hindari konflik nama
    bar_height_info = 8         # Tinggi progress bar
    bar_x_info = x + 10         # Posisi X progress bar
    bar_y_info = current_y_info # Posisi Y progress bar
    
    cv2.rectangle(img, (bar_x_info, bar_y_info), (bar_x_info + bar_width_info, bar_y_info + bar_height_info), (60, 60, 60), -1) # Latar bar
    
    progress_level = level / config.max_level # Persentase progress level
    progress_width_level = int(bar_width_info * progress_level)
    if progress_width_level > 0:
        cv2.rectangle(img, (bar_x_info, bar_y_info), (bar_x_info + progress_width_level, bar_y_info + bar_height_info), level_text_color, -1) # Isi bar

def draw_audio_control_panel(img, x, y, width, height, sound_info, sound_direction): # sound_info tidak lagi dipakai tapi argumen dipertahankan
    """Menggambar panel kontrol audio (indikator arah suara)."""
    panel_color_audio = (25, 25, 40)
    cv2.rectangle(img, (x, y), (x + width, y + height), panel_color_audio, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), config.color_schemes["secondary"], 2) # Border
    
    cv2.putText(img, "AUDIO CONTROL", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.color_schemes["text_primary"], 2)
    
    current_y_audio = y + 50 # Posisi Y saat ini
    
    # Indikator Arah Suara
    direction_indicator_colors = {
        "up": config.color_schemes["success"],
        "down": config.color_schemes["warning"],
        "neutral": config.color_schemes["text_secondary"]
    }
    cv2.putText(img, "DIRECTION:", (x + 10, current_y_audio), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.color_schemes["text_secondary"], 1)
    current_direction_color = direction_indicator_colors.get(sound_direction, config.color_schemes["text_secondary"])
    cv2.putText(img, sound_direction.upper(), (x + 10, current_y_audio + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_direction_color, 2)
    
    # Panah indikator arah
    arrow_center_x = x + width - 40
    arrow_center_y = current_y_audio + 10 # Pusatkan dengan teks "DIRECTION"
    arrow_size = 15
    
    if sound_direction == "up":
        pts = np.array([
            [arrow_center_x, arrow_center_y - arrow_size],
            [arrow_center_x - arrow_size//2, arrow_center_y + arrow_size//2],
            [arrow_center_x + arrow_size//2, arrow_center_y + arrow_size//2]
        ], np.int32)
        cv2.fillPoly(img, [pts], current_direction_color)
    elif sound_direction == "down":
        pts = np.array([
            [arrow_center_x, arrow_center_y + arrow_size],
            [arrow_center_x - arrow_size//2, arrow_center_y - arrow_size//2],
            [arrow_center_x + arrow_size//2, arrow_center_y - arrow_size//2]
        ], np.int32)
        cv2.fillPoly(img, [pts], current_direction_color)
    else: # Neutral
        cv2.circle(img, (arrow_center_x, arrow_center_y), arrow_size//2, current_direction_color, -1)

def draw_split_interface(img, width, height, fps, score, level, sound_info, sound_direction, collision_flash, level_up_flash):
    """Menggambar antarmuka terpisah: tampilan game di kiri, panel info di kanan."""
    info_panel_width_split = 250  # Lebar panel informasi, hindari konflik nama
    game_view_width_split = width - info_panel_width_split # Lebar area game
    
    # Latar belakang panel informasi (sisi kanan)
    info_bg_color_split = (15, 15, 25) # Dari config atau spesifik
    cv2.rectangle(img, (game_view_width_split, 0), (width, height), info_bg_color_split, -1)
    
    # Garis pemisah
    cv2.line(img, (game_view_width_split, 0), (game_view_width_split, height), config.color_schemes["primary"], 3)
    
    # Pengaturan panel di sisi kanan
    panel_x_split = game_view_width_split + 5
    panel_width_split_val = info_panel_width_split - 10 # Lebar efektif panel
    
    # Panel Tampilan Frekuensi (atas)
    freq_height_split = 110
    draw_frequency_display(img, panel_x_split, 10, panel_width_split_val, freq_height_split, sound_info)
    
    # Panel Info Game (tengah)
    game_info_y_split = freq_height_split + 15
    game_info_height_split = 160
    draw_game_info_panel(img, panel_x_split, game_info_y_split, panel_width_split_val, game_info_height_split, score, level, fps)
    
    # Panel Kontrol Audio (bawah)
    audio_y_split = game_info_y_split + game_info_height_split + 10
    audio_height_split = height - audio_y_split - 10 # Sisa tinggi untuk panel audio
    draw_audio_control_panel(img, panel_x_split, audio_y_split, panel_width_split_val, audio_height_split, sound_info, sound_direction)
    
    # Efek kilat hanya pada area game view
    if collision_flash > 0:
        flash_intensity_split = collision_flash / 30.0
        # Buat overlay hanya untuk area game
        flash_overlay_split = np.zeros((height, game_view_width_split, 3), dtype=np.uint8) 
        flash_overlay_split[:] = config.color_schemes["danger"]
        # Ambil slice area game dari img untuk blending
        game_area_slice = img[:, :game_view_width_split]
        cv2.addWeighted(game_area_slice, 1 - flash_intensity_split * 0.3, flash_overlay_split, flash_intensity_split * 0.3, 0, game_area_slice)
        
        if collision_flash == 30: # Saat pertama kali tabrakan
            particles.add_particles(game_view_width_split//2, height//2, config.color_schemes["danger"], 20)
    
    if level_up_flash > 0:
        flash_intensity_split = level_up_flash / 60.0
        flash_overlay_split = np.zeros((height, game_view_width_split, 3), dtype=np.uint8)
        flash_overlay_split[:] = config.color_schemes["success"]
        game_area_slice = img[:, :game_view_width_split]
        cv2.addWeighted(game_area_slice, 1 - flash_intensity_split * 0.2, flash_overlay_split, flash_intensity_split * 0.2, 0, game_area_slice)
        
        if level_up_flash == 60: # Saat pertama kali naik level
            particles.add_particles(game_view_width_split//2, 100, config.color_schemes["success"], 30)
    
    return game_view_width_split # Kembalikan lebar area game untuk logika lain

def draw_game_view_overlay(img, game_view_width, height, current_level_val):
    """Menggambar overlay minimal pada tampilan game (indikator level, instruksi)."""
    # Indikator Level (kiri atas area game)
    level_text_overlay = f"LEVEL {current_level_val}"
    (w_level_text, h_level_text), _ = cv2.getTextSize(level_text_overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    
    # Latar belakang semi-transparan untuk teks level
    overlay_level_bg = img.copy() # Salin bagian gambar yang akan ditimpa
    cv2.rectangle(overlay_level_bg, (10, 5), (10 + w_level_text + 20, 5 + h_level_text + 10), (0, 0, 0), -1) # Background hitam
    cv2.addWeighted(img, 0.7, overlay_level_bg, 0.3, 0, img) # Blend dengan alpha
    cv2.putText(img, level_text_overlay, (20, 5 + h_level_text + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.color_schemes["text_primary"], 2)
    
    # Instruksi (bawah area game)
    instruction_y_overlay = height - 60 # Posisi Y instruksi
    instructions_list = [
        "Use your voice to control the ball",
        "High pitch = UP, Low pitch = DOWN"
    ]
    
    # Hitung lebar maksimum teks instruksi untuk background
    max_instr_width = 0
    instr_line_height = 0
    for instr in instructions_list:
        (w_instr, h_instr), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        max_instr_width = max(max_instr_width, w_instr)
        if instr_line_height == 0: instr_line_height = h_instr + 5 # Ambil tinggi satu baris + padding

    # Latar belakang semi-transparan untuk instruksi
    overlay_instr_bg = img.copy()
    bg_instr_y_start = instruction_y_overlay - instr_line_height // 2 # Sesuaikan agar teks di tengah background
    bg_instr_y_end = bg_instr_y_start + len(instructions_list) * instr_line_height + 10
    cv2.rectangle(overlay_instr_bg, (10, bg_instr_y_start), (10 + max_instr_width + 20, bg_instr_y_end), (0, 0, 0), -1)
    cv2.addWeighted(img, 0.7, overlay_instr_bg, 0.3, 0, img)
    
    # Gambar teks instruksi
    for i, instr in enumerate(instructions_list):
        cv2.putText(img, instr, (20, instruction_y_overlay + i * instr_line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.color_schemes["text_secondary"], 1)

def draw_start_screen(width, height):
    """Membuat gambar untuk layar awal (start screen) game."""
    start_img = np.zeros((height, width, 3), dtype=np.uint8) # Latar hitam
    
    draw_animated_background(start_img, width, height) # Latar belakang animasi
    
    # Panel utama start screen
    panel_width_start = min(500, width - 40)
    panel_height_start = min(400, height - 40) # Sesuaikan tinggi jika perlu
    panel_x_start = (width - panel_width_start) // 2
    panel_y_start = (height - panel_height_start) // 2
    
    if panel_width_start > 10 and panel_height_start > 10: # Pastikan panel valid
        visuals_utils.draw_glassmorphism_panel(start_img, (panel_x_start, panel_y_start), 
                                       (panel_x_start + panel_width_start, panel_y_start + panel_height_start), 
                                       blur_strength=15, alpha=0.4)
    
    # Judul Game
    title_y_start = panel_y_start + 80 # Naikkan sedikit judul
    title_text_start = "SUARAKU TERBANG"
    (w_title, h_title), _ = cv2.getTextSize(title_text_start, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    
    # Efek glow animasi untuk judul
    glow_intensity_start = abs(np.sin(time.time() * 2)) # Intensitas glow berdenyut
    for i in range(3): # Beberapa layer glow
        glow_alpha_start = (0.5 - i * 0.1) * glow_intensity_start
        overlay_title = start_img.copy()
        cv2.putText(overlay_title, title_text_start, (panel_x_start + (panel_width_start - w_title) // 2, title_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, config.color_schemes["primary"], 4 + i*2) # Glow lebih tebal
        cv2.addWeighted(start_img, 1-glow_alpha_start, overlay_title, glow_alpha_start, 0, start_img)
    
    cv2.putText(start_img, title_text_start, (panel_x_start + (panel_width_start - w_title) // 2, title_y_start), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, config.color_schemes["text_primary"], 3)
    
    # Daftar Instruksi
    instructions_start = [
        "Voice-Controlled Ball Game",
        "", # Spasi
        "How to play:",
        "• High pitch sounds = Move UP",
        "• Low pitch sounds = Move DOWN", 
        "• Navigate through barriers",
        "• Reach the end to level up!",
        "• Avoid hitting barriers to keep score",
        "", # Spasi
        "Press SPACE to start"
    ]
    
    instruction_y_start_offset = title_y_start + h_title + 20 # Mulai instruksi di bawah judul
    line_height_instr = 25 # Jarak antar baris instruksi
    for instr_line in instructions_start:
        if instr_line: # Jika baris tidak kosong
            font_scale_instr = 0.6 if "Voice-Controlled" in instr_line else 0.5
            text_color_instr = config.color_schemes["text_primary"] if "Voice-Controlled" in instr_line else config.color_schemes["text_secondary"]
            thickness_instr = 2 if "Voice-Controlled" in instr_line else 1
            (w_instr_line, _), _ = cv2.getTextSize(instr_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale_instr, thickness_instr)
            cv2.putText(start_img, instr_line, (panel_x_start + (panel_width_start - w_instr_line) // 2, instruction_y_start_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_instr, text_color_instr, thickness_instr)
        instruction_y_start_offset += line_height_instr # Pindah ke baris berikutnya
    
    return start_img