"""
File Logika Inti Game.
Berisi fungsi-fungsi yang menangani mekanika dasar permainan,
seperti deteksi tabrakan antara bola dan penghalang.
"""

def check_collision_with_barriers(ball_x, ball_y, ball_width, ball_height, top_barrier_y, bottom_barrier_y, barrier_thickness):
    """
    Memeriksa apakah bola bertabrakan dengan penghalang atas atau bawah.
    Mengembalikan True jika terjadi tabrakan, False jika tidak.
    """
    ball_top = ball_y  # Sisi atas bola
    ball_bottom = ball_y + ball_height # Sisi bawah bola
    
    # Periksa tabrakan dengan penghalang atas
    # Bola dianggap bertabrakan jika sisi atasnya menyentuh atau melewati sisi bawah penghalang atas
    if ball_top <= top_barrier_y + barrier_thickness: # top_barrier_y adalah garis atas dari penghalang atas
        return True
    
    # Periksa tabrakan dengan penghalang bawah
    # Bola dianggap bertabrakan jika sisi bawahnya menyentuh atau melewati sisi atas penghalang bawah
    if ball_bottom >= bottom_barrier_y: # bottom_barrier_y adalah garis atas dari penghalang bawah
        return True
        
    return False # Tidak ada tabrakan