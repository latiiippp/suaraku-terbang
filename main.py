"""
File Utama Game "Suaraku Terbang".
Menginisialisasi game, menangani input, menjalankan loop utama game,
dan mengelola alur permainan secara keseluruhan.
"""
import cv2
import mediapipe as mp
import time
import numpy as np
import sounddevice as sd
import threading

# Impor modul-modul game
import config
import visuals_utils
import audio_processing
import particles
import visuals
import game_logic

# Cek perangkat audio yang tersedia (opsional, untuk debugging)
print(sd.query_devices())

# Muat aset gambar (kacamata dan bola)
kacamata = cv2.imread('assets/kacamata.png', cv2.IMREAD_UNCHANGED)
ball_asset = cv2.imread('assets/ball.png', cv2.IMREAD_UNCHANGED) # Ganti nama variabel agar tidak konflik

# Placeholder jika aset tidak berhasil dimuat
if kacamata is None:
    print("Warning: Could not load kacamata.png. Glasses overlay will not work.")
if ball_asset is None:
    print("Error: Could not load ball.png. Creating a placeholder.")
    ball_asset = np.zeros((50, 50, 4), dtype=np.uint8) 
    cv2.circle(ball_asset, (25, 25), 20, (0,0,255,255), -1) # Bola placeholder merah
elif ball_asset.shape[2] == 3: # Jika bola dimuat tapi 3 channel (BGR), tambahkan alpha channel
    ball_with_alpha = np.zeros((ball_asset.shape[0], ball_asset.shape[1], 4), dtype=np.uint8)
    ball_with_alpha[:,:,:3] = ball_asset
    ball_with_alpha[:,:,3] = 255 # Alpha opaque
    ball_asset = ball_with_alpha

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
# mp_visuals = mp.solutions.visuals_utils # Tidak dipakai secara eksplisit di kode ini

def main_game_loop():
    """Fungsi utama yang menjalankan loop game."""
    # Akses variabel global dari config
    # Tidak perlu deklarasi 'global' di sini karena kita memodifikasi atribut objek 'config'
    # atau menggunakan variabel yang dikelola oleh modul lain (misal config.particles)

    cap = cv2.VideoCapture(0) # Buka webcam
    
    # Atur resolusi webcam (opsional, bisa disesuaikan)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Dapatkan resolusi aktual dari webcam
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Hitung dimensi tampilan game berdasarkan aspek rasio webcam
    webcam_aspect_ratio = actual_width / actual_height
    desired_game_view_height = actual_height # Tinggi game view sama dengan tinggi webcam
    # Lebar game view disesuaikan agar proporsional dengan webcam
    desired_game_view_width = int(desired_game_view_height * webcam_aspect_ratio) 
    
    # Lebar total window = lebar game view + lebar panel info
    info_panel_width_calc = 250 # Lebar panel info (sesuaikan dengan visuals.draw_split_interface)
    window_width = desired_game_view_width + info_panel_width_calc
    window_height = actual_height # Tinggi window sama dengan tinggi webcam
    
    print(f"Window dimensions: {window_width}x{window_height}")
    
    prev_time = time.time() # Untuk perhitungan FPS
    max_score_achieved_session = 0 # Skor tertinggi dalam sesi ini, reset tiap game baru
    
    # Persiapan aset bola (resize)
    ball_display_width = 25 # Lebar bola yang diinginkan di layar
    if ball_asset is None or ball_asset.shape[1] == 0: # Jika aset bola gagal dimuat
        ball_aspect_ratio = 1 
        # Buat placeholder jika terjadi kesalahan fatal dengan aset bola
        temp_ball_placeholder = np.zeros((50,50,4), dtype=np.uint8); cv2.circle(temp_ball_placeholder,(25,25),20,(0,0,255,255),-1)
        ball_display_height = int(ball_display_width * ball_aspect_ratio)
        resized_ball_img = cv2.resize(temp_ball_placeholder, (ball_display_width, ball_display_height))
    else:
        ball_aspect_ratio = ball_asset.shape[0] / ball_asset.shape[1] # Aspek rasio asli aset bola
        ball_display_height = int(ball_display_width * ball_aspect_ratio)
        resized_ball_img = cv2.resize(ball_asset, (ball_display_width, ball_display_height))
    
    # Posisi awal bola
    center_x_ball = 0 
    center_y_ball = actual_height // 2 - resized_ball_img.shape[0] // 2

    # Jalankan thread untuk deteksi suara
    sound_processing_thread = threading.Thread(target=audio_processing.sound_thread, daemon=True)
    sound_processing_thread.start()

    frame_counter = 0 # Untuk optimasi face detection (opsional)
    collision_cooldown_timer = 0 # Cooldown setelah tabrakan (jika diperlukan)
    
    # Loop untuk layar awal (start screen)
    while not config.game_started:
        # Gambar layar awal menggunakan resolusi webcam asli
        start_screen_img = visuals.draw_start_screen(actual_width, actual_height) 
        cv2.imshow('Suaraku Terbang', start_screen_img)
        
        key_pressed = cv2.waitKey(30) & 0xFF
        if key_pressed == ord(' '): # Tekan Spasi untuk memulai
            config.game_started = True
            # Reset state game saat memulai permainan baru
            config.current_score = 0
            config.score_this_level = 0
            config.current_level = 1
            config.game_over = False
            center_x_ball = 0 # Reset posisi bola
            center_y_ball = actual_height // 2 - resized_ball_img.shape[0] // 2
            config.ball_trail.clear() # Hapus jejak bola dari game sebelumnya
            config.particles.clear()  # Hapus partikel dari game sebelumnya
            max_score_achieved_session = 0 # Reset skor tertinggi sesi
        elif key_pressed == 27: # Tekan Esc untuk keluar
            cap.release()
            cv2.destroyAllWindows()
            sd.stop() # Hentikan stream audio jika ada
            return
    
    # Inisialisasi MediaPipe Face Detection di dalam context manager
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detector:
        # Loop game utama
        while cap.isOpened() and not config.game_over:
            success, webcam_frame = cap.read() # Baca frame dari webcam
            if not success:
                print("Ignoring empty camera frame.")
                break 
            
            webcam_frame = cv2.flip(webcam_frame, 1) # Flip frame secara horizontal
            
            # Konversi frame ke RGB untuk MediaPipe
            image_rgb_for_mediapipe = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
            
            face_detection_results = None
            # Proses deteksi wajah (bisa dioptimasi dengan frame_counter % N == 0)
            if frame_counter % 1 == 0: 
                face_detection_results = face_detector.process(image_rgb_for_mediapipe)
            frame_counter += 1
            
            # Perhitungan FPS
            current_time_fps = time.time()
            time_diff_fps = current_time_fps - prev_time
            fps_value = 1 / time_diff_fps if time_diff_fps > 0 else 0
            prev_time = current_time_fps
            
            # Update skor tertinggi yang dicapai
            max_score_achieved_session = max(max_score_achieved_session, config.current_score + config.score_this_level)

            # Overlay kacamata jika wajah terdeteksi dan aset kacamata ada
            if face_detection_results and face_detection_results.detections:
                for detection in face_detection_results.detections:
                    if kacamata is not None and kacamata.shape[1] > 0: # Pastikan aset kacamata valid
                        keypoints = detection.location_data.relative_keypoints
                        left_eye = keypoints[0]  # Landmark mata kiri
                        right_eye = keypoints[1] # Landmark mata kanan

                        # Konversi koordinat relatif ke piksel
                        left_eye_x_px = int(left_eye.x * actual_width)
                        left_eye_y_px = int(left_eye.y * actual_height)
                        right_eye_x_px = int(right_eye.x * actual_width)
                        right_eye_y_px = int(right_eye.y * actual_height)

                        # Hitung pusat dan lebar mata untuk menempatkan kacamata
                        eye_center_x_px = (left_eye_x_px + right_eye_x_px) // 2
                        eye_center_y_px = (left_eye_y_px + right_eye_y_px) // 2
                        eye_width_px = int(2.5 * abs(right_eye_x_px - left_eye_x_px)) # Perbesar sedikit lebar kacamata

                        if eye_width_px > 0: 
                            scale_factor_glasses = eye_width_px / kacamata.shape[1] # Faktor skala kacamata
                            if scale_factor_glasses > 0:
                                new_glasses_width = int(kacamata.shape[1] * scale_factor_glasses)
                                new_glasses_height = int(kacamata.shape[0] * scale_factor_glasses)
                                if new_glasses_width > 0 and new_glasses_height > 0: # Pastikan dimensi valid
                                    resized_glasses = cv2.resize(kacamata, (new_glasses_width, new_glasses_height))
                                    # Hitung posisi kiri atas untuk overlay kacamata
                                    top_left_x_glasses = eye_center_x_px - new_glasses_width // 2
                                    top_left_y_glasses = eye_center_y_px - new_glasses_height // 2
                                    # Overlay kacamata ke frame webcam
                                    webcam_frame = visuals_utils.overlay_transparent(webcam_frame, resized_glasses, top_left_x_glasses, top_left_y_glasses)
            
            # Buat frame buffer untuk tampilan akhir (game view + info panel)
            final_display_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            
            # Resize frame webcam agar sesuai dengan area game view
            game_view_content = cv2.resize(webcam_frame, (desired_game_view_width, desired_game_view_height))
            # Tempatkan konten game view ke frame buffer
            final_display_frame[:desired_game_view_height, :desired_game_view_width] = game_view_content
            
            # Gambar antarmuka terpisah (panel info, dll.)
            # Fungsi ini juga mengembalikan lebar aktual dari game view yang digambar
            game_view_actual_width = visuals.draw_split_interface(
                final_display_frame, window_width, window_height, fps_value, 
                config.current_score + config.score_this_level, # Skor yang ditampilkan adalah total + level ini
                config.current_level, config.sound_info, config.sound_direction, 
                config.collision_flash, config.level_up_flash
            )
            
            # Logika pergerakan bola
            ball_speed_vertical = 3 # Kecepatan vertikal bola
            ball_speed_horizontal = 3 # Kecepatan horizontal bola

            if config.sound_direction != "neutral": # Jika ada input suara
                center_x_ball += ball_speed_horizontal # Bola bergerak ke kanan
                if config.sound_direction == "down": 
                    center_y_ball += ball_speed_vertical # Suara rendah, bola ke bawah
                elif config.sound_direction == "up": 
                    center_y_ball -= ball_speed_vertical # Suara tinggi, bola ke atas
            
            # Hitung skor untuk level ini berdasarkan posisi x bola
            if game_view_actual_width > 0:
                progress_in_level_pixels = max(0, center_x_ball) 
                # Skor proporsional dengan jarak tempuh, maks 100
                config.score_this_level = int((progress_in_level_pixels / game_view_actual_width) * 100)
                config.score_this_level = min(config.score_this_level, 100) # Batasi maksimal 100
            else:
                config.score_this_level = 0

            # Batasi posisi bola agar tetap di dalam layar game view
            center_y_ball = max(0, min(center_y_ball, actual_height - resized_ball_img.shape[0]))
            
            # Tambahkan posisi bola ke jejak
            visuals.add_ball_trail(center_x_ball + resized_ball_img.shape[1]//2, center_y_ball + resized_ball_img.shape[0]//2)
            
            # Cek jika bola mencapai ujung kanan (level selesai)
            if center_x_ball + resized_ball_img.shape[1] >= game_view_actual_width:
                if config.current_level < config.max_level: # Jika belum level maks
                    config.current_score += config.score_this_level # Tambah skor level ini ke total
                    config.score_this_level = 0 # Reset skor untuk level baru
                    config.current_level += 1   # Naik level
                    config.level_up_flash = 60  # Aktifkan efek kilat naik level
                    print(f"Level Up! Current Level: {config.current_level}, Total Score: {config.current_score}")
                else: # Jika sudah level maks
                    config.current_score += config.score_this_level 
                    config.score_this_level = 0 
                    print(f"Max Level Reached! Final Score: {config.current_score}")
                    # Bisa tambahkan logika menang di sini
                # Reset posisi bola untuk level baru atau setelah level maks
                center_x_ball = 0
                center_y_ball = actual_height // 2 - resized_ball_img.shape[0] // 2
                config.ball_trail.clear() # Hapus jejak bola

            # Pastikan bola tidak keluar dari batas kiri game view setelah reset
            center_x_ball = max(0, min(center_x_ball, game_view_actual_width - resized_ball_img.shape[1]))
            
            # Dapatkan pengaturan penghalang untuk level saat ini
            top_barrier_factor, bottom_barrier_factor = config.level_barrier_settings[config.current_level]
            top_barrier_pos_y = int(actual_height * top_barrier_factor)
            bottom_barrier_pos_y = int(actual_height * bottom_barrier_factor)
            barrier_line_thickness = 3
            
            # Cek tabrakan dengan penghalang
            if collision_cooldown_timer <= 0: # Hanya cek jika tidak dalam cooldown
                if game_logic.check_collision_with_barriers(
                    center_x_ball, center_y_ball, resized_ball_img.shape[1], 
                    resized_ball_img.shape[0], top_barrier_pos_y, bottom_barrier_pos_y, barrier_line_thickness
                ):
                    final_score_at_collision = config.current_score + config.score_this_level
                    print(f"Collision! Game Over. Final Score: {final_score_at_collision}")
                    config.collision_flash = 30 # Aktifkan efek kilat tabrakan
                    config.game_over = True     # Set status game over
            else:
                collision_cooldown_timer -= 1 # Kurangi timer cooldown
            
            # Update timer efek kilat
            if config.collision_flash > 0: config.collision_flash -= 1
            if config.level_up_flash > 0: config.level_up_flash -= 1
            
            particles.update_particles() # Update semua partikel
            
            # Gambar elemen-elemen game pada slice game view dari frame buffer
            game_view_slice_to_draw_on = final_display_frame[:, :game_view_actual_width]
            visuals.draw_ball_trail(game_view_slice_to_draw_on)
            game_view_slice_to_draw_on = visuals_utils.overlay_transparent(game_view_slice_to_draw_on, resized_ball_img, center_x_ball, center_y_ball)
            visuals.draw_modern_barriers(game_view_slice_to_draw_on, game_view_actual_width, window_height, top_barrier_pos_y, bottom_barrier_pos_y, barrier_line_thickness)
            particles.draw_particles(game_view_slice_to_draw_on)
            visuals.draw_game_view_overlay(game_view_slice_to_draw_on, game_view_actual_width, window_height, config.current_level)
            
            cv2.imshow('Suaraku Terbang', final_display_frame) # Tampilkan frame akhir
            
            key_pressed_ingame = cv2.waitKey(5) & 0xFF 
            if key_pressed_ingame == 27 or cv2.getWindowProperty('Suaraku Terbang', cv2.WND_PROP_VISIBLE) < 1: # Esc atau tutup window
                break # Keluar dari loop game
        
        # Setelah loop game utama selesai (game over atau keluar)
        if config.game_over:
            final_score_to_display = config.current_score + config.score_this_level
            print(f"Final Score on Game Over Screen: {final_score_to_display}") 
            # Gambar layar game over menggunakan resolusi webcam asli
            game_over_screen_img = visuals.draw_modern_game_over(actual_width, actual_height, final_score_to_display, max_score_achieved_session)
            cv2.imshow('Suaraku Terbang', game_over_screen_img)
            cv2.waitKey(0) # Tunggu input keyboard apapun untuk menutup
                
    # Bebaskan resource webcam dan tutup semua window OpenCV
    cap.release()
    cv2.destroyAllWindows()
    sd.stop() # Hentikan stream audio jika ada

if __name__ == "__main__":
    main_game_loop()