import cv2
import mediapipe as mp
import time
import numpy as np
import sounddevice as sd
import threading

print(sd.query_devices())

kacamata = cv2.imread('assets/kacamata.png', cv2.IMREAD_UNCHANGED)
ball = cv2.imread('assets/ball.png', cv2.IMREAD_UNCHANGED)

# Placeholder if assets are not loaded
if kacamata is None:
    print("Warning: Could not load kacamata.png. Glasses overlay will not work.")
if ball is None:
    print("Error: Could not load ball.png. Creating a placeholder.")
    ball = np.zeros((50, 50, 4), dtype=np.uint8) 
    cv2.circle(ball, (25, 25), 20, (0,0,255,255), -1) # Red ball placeholder
elif ball.shape[2] == 3: # If ball is loaded but 3-channel, add alpha
    ball_with_alpha = np.zeros((ball.shape[0], ball.shape[1], 4), dtype=np.uint8)
    ball_with_alpha[:,:,:3] = ball
    ball_with_alpha[:,:,3] = 255
    ball = ball_with_alpha


# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x + w > bw: w = bw - x
    if y + h > bh: h = bh - y
    if x < 0: 
        overlay = overlay[:, -x:] # Crop overlay from left
        w += x # Adjust width
        x = 0
    if y < 0: 
        overlay = overlay[-y:, :] # Crop overlay from top
        h += y # Adjust height
        y = 0

    if w <= 0 or h <= 0 or overlay is None:
        return background

    overlay_crop = overlay[:h, :w]
    
    if overlay_crop.shape[2] < 4:
        print("Warning: Overlay image is not RGBA. Skipping overlay.")
        return background

    bg_region = background[y:y+h, x:x+w]
    
    if bg_region.shape[0] != overlay_crop.shape[0] or bg_region.shape[1] != overlay_crop.shape[1]:
        return background

    alpha_overlay = overlay_crop[:, :, 3:] / 255.0
    color_overlay = overlay_crop[:, :, :3]
    
    blended_color = color_overlay * alpha_overlay + bg_region[:,:,:3] * (1.0 - alpha_overlay)
    
    background[y:y+h, x:x+w, :3] = blended_color.astype(np.uint8)

    if background.shape[2] == 4: # If background itself has an alpha channel
        alpha_bg_region = bg_region[:, :, 3:] / 255.0
        new_alpha_bg = alpha_overlay + alpha_bg_region * (1.0 - alpha_overlay)
        background[y:y+h, x:x+w, 3] = (new_alpha_bg * 255).astype(np.uint8)

    return background

sound_direction = "neutral"
last_movement_direction = "neutral"
sound_info = {"bass_energy": 0, "treble_energy": 0, "dominant_freq": 0}

# Level settings
current_level = 1
max_level = 5
# Scoring system
current_score = 100
score_per_level = 50
barrier_penalty = 10
game_over = False
game_started = False

# UI Animation variables
collision_flash = 0
level_up_flash = 0
ui_animations = {"score_pulse": 0, "level_glow": 0, "background_wave": 0, "particle_time": 0}

# Particle system
particles = []
ball_trail = []
max_trail_length = 15

# Color schemes
color_schemes = {
    "primary": (100, 150, 255),
    "secondary": (50, 200, 150),
    "accent": (255, 100, 150),
    "background_dark": (15, 15, 25),
    "background_light": (25, 25, 35),
    "text_primary": (255, 255, 255),
    "text_secondary": (180, 180, 200),
    "success": (100, 255, 100),
    "warning": (255, 200, 100),
    "danger": (255, 100, 100)
}

# (top_factor, bottom_factor) for barrier y-positions relative to screen height
level_barrier_settings = {
    1: (0.30, 0.70),  # Gap: 40% of height
    2: (0.35, 0.65),  # Gap: 30% of height
    3: (0.40, 0.60),  # Gap: 20% of height
    4: (0.425, 0.575), # Gap: 15% of height
    5: (0.45, 0.55)   # Gap: 10% of height
}

def detect_sound_direction(duration=0.1, sample_rate=44100):
    global sound_info, last_movement_direction
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=4, dtype='float32', device=1)
    sd.wait()
    audio_data = recording.flatten()

    fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    magnitudes = np.abs(fft)

    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitudes = magnitudes[:len(magnitudes)//2]

    current_determined_direction = "neutral" 

    if len(positive_magnitudes) == 0 or np.sum(positive_magnitudes) == 0:
        sound_info["bass_energy"] = 0
        sound_info["treble_energy"] = 0
        sound_info["dominant_freq"] = 0
    else:
        max_magnitude_idx = np.argmax(positive_magnitudes)
        dominant_freq = positive_freqs[max_magnitude_idx]
        
        sound_info["bass_energy"] = np.sum(positive_magnitudes[(positive_freqs >= 1) & (positive_freqs <= 150)])
        sound_info["treble_energy"] = np.sum(positive_magnitudes[(positive_freqs > 150)])
        sound_info["dominant_freq"] = dominant_freq

        if dominant_freq == 0:
            current_determined_direction = "neutral"
        elif 1 <= dominant_freq <= 150: 
            current_determined_direction = "down"
        elif dominant_freq > 150: 
            current_determined_direction = "up"
            
    last_movement_direction = current_determined_direction
    return current_determined_direction

def sound_thread():
    global sound_direction
    while True:
        sound_direction = detect_sound_direction()
        time.sleep(0.01) 

def check_collision_with_barriers(ball_x, ball_y, ball_width, ball_height, top_barrier_y, bottom_barrier_y, barrier_thickness):
    """Check if ball collides with top or bottom barriers"""
    ball_top = ball_y
    ball_bottom = ball_y + ball_height
    
    # Check collision with top barrier
    if ball_top <= top_barrier_y + barrier_thickness:
        return True
    
    # Check collision with bottom barrier  
    if ball_bottom >= bottom_barrier_y:
        return True
        
    return False

def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=10, filled=False):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    if filled:
        # Create mask for rounded rectangle
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1+radius, y1), (x2-radius, y2), 255, -1)
        cv2.rectangle(mask, (x1, y1+radius), (x2, y2-radius), 255, -1)
        cv2.circle(mask, (x1+radius, y1+radius), radius, 255, -1)
        cv2.circle(mask, (x2-radius, y1+radius), radius, 255, -1)
        cv2.circle(mask, (x1+radius, y2-radius), radius, 255, -1)
        cv2.circle(mask, (x2-radius, y2-radius), radius, 255, -1)
        
        # Apply color to masked area
        img[mask == 255] = color
    else:
        # Draw outline
        cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
        cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
        cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
        cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_gradient_panel(img, pt1, pt2, color1, color2, alpha=0.8):
    """Draw a gradient panel with transparency"""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    
    # Create gradient
    gradient = np.linspace(0, 1, y2-y1).reshape(-1, 1)
    gradient_rgb = gradient * np.array(color2) + (1-gradient) * np.array(color1)
    
    for i in range(y2-y1):
        color = tuple(map(int, gradient_rgb[i]))
        cv2.rectangle(overlay, (x1, y1+i), (x2, y1+i+1), color, -1)
    
    # Apply transparency
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_modern_hud(img, width, height, fps, score, level, sound_info, sound_direction, collision_flash, level_up_flash):
    """Draw modern HUD with panels and animations"""
    global ui_animations
    
    # Update animations
    ui_animations["score_pulse"] = (ui_animations["score_pulse"] + 1) % 60
    ui_animations["level_glow"] = (ui_animations["level_glow"] + 1) % 120
    
    # Top panel - Score and Level
    panel_height = 80
    draw_gradient_panel(img, (0, 0), (width, panel_height), (30, 30, 40), (20, 20, 30), 0.85)
    
    # Score panel with pulse effect
    score_pulse = abs(np.sin(ui_animations["score_pulse"] * 0.1)) * 20
    score_color = (50 + int(score_pulse), 255, 50 + int(score_pulse))
    
    cv2.putText(img, "SCORE", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(img, f"{score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 2)
    
    # Level panel with glow effect
    level_glow = abs(np.sin(ui_animations["level_glow"] * 0.05)) * 30
    level_color = (50, 150 + int(level_glow), 255)
    
    cv2.putText(img, "LEVEL", (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(img, f"{level}/{max_level}", (width - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, level_color, 2)
    
    # Progress bar for level
    bar_width = 100
    bar_height = 8
    bar_x = width - 130
    bar_y = 60
    
    # Background bar
    draw_rounded_rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), filled=True, radius=4)
    
    # Progress bar
    progress = level / max_level
    progress_width = int(bar_width * progress)
    if progress_width > 0:
        draw_rounded_rectangle(img, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), level_color, filled=True, radius=4)
    
    # Side panel - Audio info
    panel_width = 250
    panel_x = 10
    panel_y = 90
    panel_bottom = height - 20
    
    draw_gradient_panel(img, (panel_x, panel_y), (panel_x + panel_width, panel_bottom), (25, 25, 35), (15, 15, 25), 0.85)
    draw_rounded_rectangle(img, (panel_x, panel_y), (panel_x + panel_width, panel_bottom), (100, 100, 120), 2, 8)
    
    # Audio info with modern styling
    y_offset = panel_y + 25
    
    cv2.putText(img, "AUDIO ANALYSIS", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 30
    
    # Frequency display with color coding
    freq = int(sound_info['dominant_freq'])
    freq_color = (0, 255, 255) if freq > 150 else (255, 255, 0) if freq > 0 else (128, 128, 128)
    cv2.putText(img, f"Frequency: {freq} Hz", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, freq_color, 1)
    y_offset += 25
    
    # Bass energy bar
    bass_energy = min(int(sound_info['bass_energy'] / 100), 100)
    cv2.putText(img, "Bass:", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    bar_bg = (panel_x + 60, y_offset - 10, panel_x + 200, y_offset)
    cv2.rectangle(img, (bar_bg[0], bar_bg[1]), (bar_bg[2], bar_bg[3]), (50, 50, 50), -1)
    if bass_energy > 0:
        bar_fill_width = int((bar_bg[2] - bar_bg[0]) * bass_energy / 100)
        cv2.rectangle(img, (bar_bg[0], bar_bg[1]), (bar_bg[0] + bar_fill_width, bar_bg[3]), (0, 255, 255), -1)
    y_offset += 25
    
    # Treble energy bar
    treble_energy = min(int(sound_info['treble_energy'] / 100), 100)
    cv2.putText(img, "Treble:", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    bar_bg = (panel_x + 60, y_offset - 10, panel_x + 200, y_offset)
    cv2.rectangle(img, (bar_bg[0], bar_bg[1]), (bar_bg[2], bar_bg[3]), (50, 50, 50), -1)
    if treble_energy > 0:
        bar_fill_width = int((bar_bg[2] - bar_bg[0]) * treble_energy / 100)
        cv2.rectangle(img, (bar_bg[0], bar_bg[1]), (bar_bg[0] + bar_fill_width, bar_bg[3]), (255, 0, 255), -1)
    y_offset += 30
    
    # Direction indicator with arrows
    direction_colors = {
        "up": (0, 255, 0),
        "down": (255, 100, 0),
        "neutral": (128, 128, 128)
    }
    
    cv2.putText(img, "DIRECTION:", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    direction_color = direction_colors.get(sound_direction, (128, 128, 128))
    
    # Draw direction arrow
    arrow_center = (panel_x + 120, y_offset - 8)
    if sound_direction == "up":
        cv2.arrowedLine(img, (arrow_center[0], arrow_center[1] + 10), (arrow_center[0], arrow_center[1] - 10), direction_color, 3)
    elif sound_direction == "down":
        cv2.arrowedLine(img, (arrow_center[0], arrow_center[1] - 10), (arrow_center[0], arrow_center[1] + 10), direction_color, 3)
    else:
        cv2.circle(img, arrow_center, 5, direction_color, -1)
    
    cv2.putText(img, sound_direction.upper(), (panel_x + 140, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_color, 1)
    
    # FPS display
    cv2.putText(img, f"FPS: {int(fps)}", (panel_x + 15, panel_bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    
    # Collision flash effect
    if collision_flash > 0:
        flash_alpha = collision_flash / 30.0
        flash_overlay = img.copy()
        cv2.rectangle(flash_overlay, (0, 0), (width, height), (0, 0, 255), -1)
        cv2.addWeighted(img, 1-flash_alpha*0.3, flash_overlay, flash_alpha*0.3, 0, img)
    
    # Level up flash effect
    if level_up_flash > 0:
        flash_alpha = level_up_flash / 60.0
        flash_overlay = img.copy()
        cv2.rectangle(flash_overlay, (0, 0), (width, height), (0, 255, 0), -1)
        cv2.addWeighted(img, 1-flash_alpha*0.2, flash_overlay, flash_alpha*0.2, 0, img)

def draw_modern_barriers(img, width, height, top_y, bottom_y, thickness):
    """Draw modern looking barriers with glow effect"""
    barrier_color = (100, 150, 255)
    glow_color = (50, 100, 200)
    
    # Draw glow effect
    for i in range(3):
        glow_thickness = thickness + (3-i) * 2
        alpha = 0.3 - i * 0.1
        overlay = img.copy()
        cv2.line(overlay, (0, top_y), (width, top_y), glow_color, glow_thickness)
        cv2.line(overlay, (0, bottom_y), (width, bottom_y), glow_color, glow_thickness)
        cv2.addWeighted(img, 1-alpha, overlay, alpha, 0, img)
    
    # Draw main barriers
    cv2.line(img, (0, top_y), (width, top_y), barrier_color, thickness)
    cv2.line(img, (0, bottom_y), (width, bottom_y), barrier_color, thickness)

def draw_modern_game_over(width, height, final_score, max_score_achieved):
    """Create a modern game over screen"""
    game_over_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background gradient
    draw_gradient_panel(game_over_img, (0, 0), (width, height), (20, 20, 30), (40, 40, 60), 1.0)
    
    # Main panel
    panel_width = 400
    panel_height = 300
    panel_x = (width - panel_width) // 2
    panel_y = (height - panel_height) // 2
    
    draw_gradient_panel(game_over_img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                       (60, 60, 80), (40, 40, 60), 0.95)
    draw_rounded_rectangle(game_over_img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                          (150, 150, 200), 3, 15)
    
    # Game Over text with glow
    text_center_x = width // 2 - 120
    text_y = panel_y + 80
    
    # Glow effect for title
    for i in range(3):
        glow_size = 2.2 + i * 0.2
        glow_alpha = 0.3 - i * 0.1
        overlay = game_over_img.copy()
        cv2.putText(overlay, "GAME OVER", (text_center_x - i*2, text_y + i*2), 
                   cv2.FONT_HERSHEY_SIMPLEX, glow_size, (100, 100, 255), 4)
        cv2.addWeighted(game_over_img, 1-glow_alpha, overlay, glow_alpha, 0, game_over_img)
    
    cv2.putText(game_over_img, "GAME OVER", (text_center_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Score display
    score_y = text_y + 80
    cv2.putText(game_over_img, f"Final Score: {final_score}", (text_center_x - 20, score_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    
    # Instructions
    instruction_y = score_y + 60
    cv2.putText(game_over_img, "Press any key to exit", (text_center_x - 40, instruction_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return game_over_img

class Particle:
    def __init__(self, x, y, vx, vy, color, size, life):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # gravity
        self.life -= 1
        
    def is_alive(self):
        return self.life > 0
        
    def draw(self, img):
        if self.is_alive():
            alpha = self.life / self.max_life
            size = int(self.size * alpha)
            if size > 0:
                cv2.circle(img, (int(self.x), int(self.y)), size, self.color, -1)

def add_particles(x, y, color, count=10):
    global particles
    for _ in range(count):
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(-5, -1)
        size = np.random.randint(2, 6)
        life = np.random.randint(20, 40)
        particles.append(Particle(x, y, vx, vy, color, size, life))

def update_particles():
    global particles
    particles = [p for p in particles if p.is_alive()]
    for particle in particles:
        particle.update()

def draw_particles(img):
    for particle in particles:
        particle.draw(img)

def add_ball_trail(x, y):
    global ball_trail
    ball_trail.append((x, y))
    if len(ball_trail) > max_trail_length:
        ball_trail.pop(0)

def draw_ball_trail(img):
    if len(ball_trail) < 2:
        return
    
    for i in range(1, len(ball_trail)):
        alpha = i / len(ball_trail)
        thickness = int(3 * alpha)
        if thickness > 0:
            color = tuple(int(c * alpha) for c in color_schemes["accent"])
            cv2.line(img, ball_trail[i-1], ball_trail[i], color, thickness)

def draw_glassmorphism_panel(img, pt1, pt2, blur_strength=15, alpha=0.3):
    """Create glassmorphism effect panel"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Ensure coordinates are valid and within bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    # Check if region is valid
    if x2 <= x1 or y2 <= y1:
        return
    
    # Ensure blur strength is odd and at least 1
    blur_strength = max(1, blur_strength)
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Extract background region
    bg_region = img[y1:y2, x1:x2].copy()
    
    # Check if region is large enough for blur
    if bg_region.shape[0] < blur_strength or bg_region.shape[1] < blur_strength:
        # Use smaller blur for small regions
        blur_strength = min(bg_region.shape[0], bg_region.shape[1])
        if blur_strength < 3:
            blur_strength = 1
        elif blur_strength % 2 == 0:
            blur_strength -= 1
    
    # Apply blur only if region is valid
    if bg_region.shape[0] > 0 and bg_region.shape[1] > 0:
        blurred = cv2.GaussianBlur(bg_region, (blur_strength, blur_strength), 0)
        
        # Create white overlay for glass effect
        overlay = np.ones_like(blurred) * 40
        overlay[:, :, 0] = 60  # Slight blue tint
        overlay[:, :, 1] = 60
        overlay[:, :, 2] = 80
        
        # Blend with blurred background
        glass_effect = cv2.addWeighted(blurred, 1-alpha, overlay, alpha, 0)
        
        # Apply back to image
        img[y1:y2, x1:x2] = glass_effect
        
        # Add border glow
        border_color = color_schemes["primary"]
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
        
        # Add inner glow
        inner_overlay = img.copy()
        cv2.rectangle(inner_overlay, (x1+2, y1+2), (x2-2, y2-2), border_color, 1)
        cv2.addWeighted(img, 0.9, inner_overlay, 0.1, 0, img)

def draw_animated_background(img, width, height):
    """Draw animated background with waves and patterns"""
    global ui_animations
    
    ui_animations["background_wave"] = (ui_animations["background_wave"] + 1) % 360
    
    # Create wave pattern
    wave_offset = ui_animations["background_wave"] * 0.02
    
    for y in range(0, height, 20):
        for x in range(0, width, 20):
            wave = np.sin((x * 0.01) + wave_offset) * np.cos((y * 0.01) + wave_offset)
            intensity = int(abs(wave) * 15)
            
            color = (
                color_schemes["background_dark"][0] + intensity,
                color_schemes["background_dark"][1] + intensity,
                color_schemes["background_dark"][2] + intensity
            )
            
            cv2.circle(img, (x, y), 2, color, -1)

def draw_sound_visualizer(img, x, y, width, height, sound_info):
    """Draw real-time sound visualization"""
    # Frequency bars
    bar_count = 20
    bar_width = width // bar_count
    
    # Simulate frequency spectrum
    bass_ratio = min(sound_info['bass_energy'] / 1000, 1.0)
    treble_ratio = min(sound_info['treble_energy'] / 1000, 1.0)
    
    for i in range(bar_count):
        # Create varying heights based on frequency
        if i < bar_count // 2:
            bar_height = int(height * bass_ratio * (0.5 + 0.5 * np.sin(i * 0.5 + time.time() * 2)))
        else:
            bar_height = int(height * treble_ratio * (0.5 + 0.5 * np.sin(i * 0.5 + time.time() * 3)))
        
        bar_x = x + i * bar_width
        bar_y = y + height - bar_height
        
        # Color gradient from bass (blue) to treble (red)
        ratio = i / bar_count
        color = (
            int(100 + 155 * ratio),
            int(150 - 100 * ratio),
            int(255 - 155 * ratio)
        )
        
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width - 2, y + height), color, -1)

def draw_advanced_hud(img, width, height, fps, score, level, sound_info, sound_direction, collision_flash, level_up_flash):
    """Advanced HUD with glassmorphism and animations"""
    global ui_animations
    
    # Update animations
    ui_animations["score_pulse"] = (ui_animations["score_pulse"] + 1) % 60
    ui_animations["level_glow"] = (ui_animations["level_glow"] + 1) % 120
    ui_animations["particle_time"] = (ui_animations["particle_time"] + 1) % 360
    
    # Top HUD panel with glassmorphism
    panel_height = 100
    # Ensure panel dimensions are valid
    if width > 10 and panel_height > 10:
        draw_glassmorphism_panel(img, (0, 0), (width, panel_height), 15, 0.4)
    
    # Animated score with glow effect
    score_pulse = abs(np.sin(ui_animations["score_pulse"] * 0.1))
    score_glow = int(50 + score_pulse * 50)
    
    # Score with multiple glow layers
    for i in range(3):
        glow_alpha = (0.3 - i * 0.1) * score_pulse
        glow_offset = i * 2
        overlay = img.copy()
        cv2.putText(overlay, f"{score}", (30 + glow_offset, 60 + glow_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color_schemes["success"], 3)
        cv2.addWeighted(img, 1-glow_alpha, overlay, glow_alpha, 0, img)
    
    cv2.putText(img, "SCORE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_schemes["text_secondary"], 2)
    cv2.putText(img, f"{score}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color_schemes["text_primary"], 3)
    
    # Animated level indicator
    level_progress = level / max_level
    level_glow = abs(np.sin(ui_animations["level_glow"] * 0.05))
    
    # Level text
    cv2.putText(img, "LEVEL", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_schemes["text_secondary"], 2)
    cv2.putText(img, f"{level}", (width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color_schemes["primary"], 3)
    
    # Circular progress indicator for level
    center = (width - 100, 50)
    radius = 30
    
    # Background circle
    cv2.circle(img, center, radius, color_schemes["background_light"], 3)
    
    # Progress arc
    angle = int(360 * level_progress)
    if angle > 0:
        # Create arc points
        arc_color = tuple(int(c + level_glow * 50) for c in color_schemes["secondary"])
        for i in range(angle):
            x = int(center[0] + radius * np.cos(np.radians(i - 90)))
            y = int(center[1] + radius * np.sin(np.radians(i - 90)))
            cv2.circle(img, (x, y), 3, arc_color, -1)
    
    # Side panel for audio visualization
    panel_x, panel_y = 20, 120
    panel_width, panel_height = 300, height - 140
    
    # Ensure panel dimensions are valid
    if panel_width > 10 and panel_height > 10 and panel_x + panel_width <= width and panel_y + panel_height <= height:
        draw_glassmorphism_panel(img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 15, 0.35)
    
    # Audio section header
    cv2.putText(img, "AUDIO CONTROL", (panel_x + 20, panel_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_schemes["text_primary"], 2)
    
    # Sound visualizer
    visualizer_y = panel_y + 50
    draw_sound_visualizer(img, panel_x + 20, visualizer_y, panel_width - 40, 80, sound_info)
    
    # Direction indicator with smooth animation
    direction_y = visualizer_y + 100
    direction_colors = {
        "up": color_schemes["success"],
        "down": color_schemes["warning"],
        "neutral": color_schemes["text_secondary"]
    }
    
    current_color = direction_colors.get(sound_direction, color_schemes["text_secondary"])
    
    # Animated direction arrow
    arrow_center = (panel_x + 150, direction_y + 20)
    arrow_size = 15
    
    if sound_direction == "up":
        pts = np.array([
            [arrow_center[0], arrow_center[1] - arrow_size],
            [arrow_center[0] - arrow_size//2, arrow_center[1] + arrow_size//2],
            [arrow_center[0] + arrow_size//2, arrow_center[1] + arrow_size//2]
        ], np.int32)
        cv2.fillPoly(img, [pts], current_color)
    elif sound_direction == "down":
        pts = np.array([
            [arrow_center[0], arrow_center[1] + arrow_size],
            [arrow_center[0] - arrow_size//2, arrow_center[1] - arrow_size//2],
            [arrow_center[0] + arrow_size//2, arrow_center[1] - arrow_size//2]
        ], np.int32)
        cv2.fillPoly(img, [pts], current_color)
    else:
        cv2.circle(img, arrow_center, arrow_size//2, current_color, -1)
    
    cv2.putText(img, f"DIRECTION: {sound_direction.upper()}", 
               (panel_x + 20, direction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 1)
    
    # FPS and performance info
    cv2.putText(img, f"FPS: {int(fps)}", (panel_x + 20, panel_y + panel_height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_schemes["text_secondary"], 1)
    
    # Flash effects with smooth transitions
    if collision_flash > 0:
        flash_intensity = collision_flash / 30.0
        flash_overlay = np.zeros_like(img)
        flash_overlay[:] = color_schemes["danger"]
        cv2.addWeighted(img, 1 - flash_intensity * 0.3, flash_overlay, flash_intensity * 0.3, 0, img)
        
        # Add particles on collision
        if collision_flash == 30:  # First frame of collision
            add_particles(width//2, height//2, color_schemes["danger"], 20)
    
    if level_up_flash > 0:
        flash_intensity = level_up_flash / 60.0
        flash_overlay = np.zeros_like(img)
        flash_overlay[:] = color_schemes["success"]
        cv2.addWeighted(img, 1 - flash_intensity * 0.2, flash_overlay, flash_intensity * 0.2, 0, img)
        
        # Add celebration particles
        if level_up_flash == 60:  # First frame of level up
            add_particles(width//2, 100, color_schemes["success"], 30)

def draw_start_screen(width, height):
    """Create an attractive start screen"""
    start_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Animated background
    draw_animated_background(start_img, width, height)
    
    # Main panel
    panel_width = min(500, width - 40)  # Ensure panel fits in screen
    panel_height = min(400, height - 40)
    panel_x = (width - panel_width) // 2
    panel_y = (height - panel_height) // 2
    
    # Ensure panel dimensions are valid
    if panel_width > 10 and panel_height > 10:
        draw_glassmorphism_panel(start_img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 15, 0.4)
    
    # Title with animated glow
    title_y = panel_y + 100
    title_text = "SUARAKU TERBANG"
    
    # Animated title glow
    glow_intensity = abs(np.sin(time.time() * 2))
    for i in range(5):
        glow_alpha = (0.5 - i * 0.1) * glow_intensity
        glow_offset = i * 3
        overlay = start_img.copy()
        cv2.putText(overlay, title_text, (panel_x + 50 + glow_offset, title_y + glow_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_schemes["primary"], 4)
        cv2.addWeighted(start_img, 1-glow_alpha, overlay, glow_alpha, 0, start_img)
    
    cv2.putText(start_img, title_text, (panel_x + 50, title_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_schemes["text_primary"], 3)
    
    # Instructions
    instructions = [
        "Use your voice to control the ball:",
        "• High pitch sounds = Move UP",
        "• Low pitch sounds = Move DOWN", 
        "• Navigate through barriers",
        "• Reach the end to level up!",
        "",
        "Press SPACE to start"
    ]
    
    instruction_y = title_y + 80
    for instruction in instructions:
        if instruction:
            cv2.putText(start_img, instruction, (panel_x + 50, instruction_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_schemes["text_secondary"], 1)
        instruction_y += 30
    
    return start_img

def main():
    global current_level, current_score, game_over, collision_flash, level_up_flash, game_started

    cap = cv2.VideoCapture(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    prev_time = time.time()
    max_score_achieved = current_score
    
    ball_width = 25 
    if ball is None or ball.shape[1] == 0: 
        ball_aspect_ratio = 1 
        temp_ball_img = np.zeros((50, 50, 4), dtype=np.uint8)
        cv2.circle(temp_ball_img, (25, 25), 20, (0,0,255,255), -1)
        ball_height = int(ball_width * ball_aspect_ratio)
        resized_ball = cv2.resize(temp_ball_img, (ball_width, ball_height))
    else:
        ball_aspect_ratio = ball.shape[0] / ball.shape[1]
        ball_height = int(ball_width * ball_aspect_ratio)
        resized_ball = cv2.resize(ball, (ball_width, ball_height))
    
    center_x = 0 
    center_y = actual_height // 2 - resized_ball.shape[0] // 2

    threading.Thread(target=sound_thread, daemon=True).start()

    frame_count = 0
    collision_cooldown = 0
    
    # Show start screen
    while not game_started:
        start_img = draw_start_screen(actual_width, actual_height)
        cv2.imshow('MediaPipe Face Detection', start_img)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            game_started = True
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened() and not game_over:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break 
            
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Draw animated background
            draw_animated_background(image, actual_width, actual_height)
            
            results = None
            if frame_count % 1 == 0:
                results = face_detection.process(image_rgb)
            frame_count += 1
            
            current_time = time.time()
            time_diff = current_time - prev_time
            fps = 1 / time_diff if time_diff > 0 else 0
            prev_time = current_time
            
            max_score_achieved = max(max_score_achieved, current_score)

            if results and results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    
                    if kacamata is not None and kacamata.shape[1] > 0:
                        keypoints = detection.location_data.relative_keypoints
                        left_eye = keypoints[0]
                        right_eye = keypoints[1]

                        left_eye_x = int(left_eye.x * actual_width)
                        left_eye_y = int(left_eye.y * actual_height)
                        right_eye_x = int(right_eye.x * actual_width)
                        right_eye_y = int(right_eye.y * actual_height)

                        eye_center_x = (left_eye_x + right_eye_x) // 2
                        eye_center_y = (left_eye_y + right_eye_y) // 2
                        eye_width = int(2.5 * abs(right_eye_x - left_eye_x))

                        scale_factor = eye_width / kacamata.shape[1]
                        new_glasses = cv2.resize(kacamata, (0, 0), fx=scale_factor, fy=scale_factor)

                        gh, gw = new_glasses.shape[:2]
                        top_left_x = eye_center_x - gw // 2
                        top_left_y = eye_center_y - gh // 2
                        image = overlay_transparent(image, new_glasses, top_left_x, top_left_y)
            
            current_sound_direction_val = sound_direction 
            
            ball_speed_vertical = 4
            ball_speed_horizontal = 2 

            if current_sound_direction_val != "neutral":
                center_x += ball_speed_horizontal
                if current_sound_direction_val == "down": 
                    center_y += ball_speed_vertical
                elif current_sound_direction_val == "up": 
                    center_y -= ball_speed_vertical
            
            center_y = max(0, min(center_y, actual_height - resized_ball.shape[0]))
            
            # Add ball trail
            add_ball_trail(center_x + resized_ball.shape[1]//2, center_y + resized_ball.shape[0]//2)
            
            # Level up logic
            if center_x + resized_ball.shape[1] >= actual_width:
                if current_level < max_level:
                    current_level += 1
                    current_score += score_per_level  # Bonus for leveling up
                    level_up_flash = 60  # Start level up animation
                    print(f"Level Up! Current Level: {current_level}, Score: {current_score}")
                else:
                    print("Max Level Reached! Resetting to Level 1 or staying at max.")
                center_x = 0
                center_y = actual_height // 2 - resized_ball.shape[0] // 2

            center_x = max(0, min(center_x, actual_width - resized_ball.shape[1])) 
            
            # Calculate barrier positions
            top_barrier_factor, bottom_barrier_factor = level_barrier_settings[current_level]
            top_barrier_y = int(actual_height * top_barrier_factor)
            bottom_barrier_y = int(actual_height * bottom_barrier_factor)
            barrier_thickness = 3
            
            # Collision detection with cooldown
            if collision_cooldown <= 0:
                if check_collision_with_barriers(center_x, center_y, resized_ball.shape[1], 
                                                resized_ball.shape[0], top_barrier_y, bottom_barrier_y, barrier_thickness):
                    current_score -= barrier_penalty
                    collision_cooldown = 30  # 30 frames cooldown
                    collision_flash = 30  # Start collision animation
                    print(f"Collision! Score reduced to: {current_score}")
                    
                    # Check game over
                    if current_score <= 0:
                        game_over = True
                        print("Game Over! Score reached zero.")
            else:
                collision_cooldown -= 1
            
            # Update and draw particles
            update_particles()
            
            # Draw ball trail
            draw_ball_trail(image)
            
            image = overlay_transparent(image, resized_ball, center_x, center_y)
            
            # Draw modern barriers
            draw_modern_barriers(image, actual_width, actual_height, top_barrier_y, bottom_barrier_y, barrier_thickness)
            
            # Draw particles
            draw_particles(image)
            
            # Draw advanced HUD
            draw_advanced_hud(image, actual_width, actual_height, fps, current_score, current_level, 
                            sound_info, sound_direction, collision_flash, level_up_flash)
            
            cv2.imshow('MediaPipe Face Detection', image)
            
            key = cv2.waitKey(5) & 0xFF 
            if key == 27 or cv2.getWindowProperty('MediaPipe Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # Game over screen
        if game_over:
            print(f"Final Score: {current_score}")
            game_over_img = draw_modern_game_over(actual_width, actual_height, current_score, max_score_achieved)
            cv2.imshow('MediaPipe Face Detection', game_over_img)
            cv2.waitKey(0)
                
    cap.release()
    cv2.destroyAllWindows()
    sd.stop()

if __name__ == "__main__":
    main()