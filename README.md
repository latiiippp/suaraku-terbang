# Filter Tiktok Suaraku Terbang

## Deskripsi

Selamat datang di **"Suaraku Terbang"**! Sebuah permainan interaktif yang terinspirasi dari filter TikTok, di mana suara Anda menjadi pengendali utama. Dalam permainan ini, Anda akan memandu sebuah bola tetap bergerak tanpa mengenai penghalang hanya dengan menggunakan tinggi rendahnya nada suara Anda.

**Mekanisme Inti:**
Permainan ini mendeteksi frekuensi suara Anda:

- **Nada Tinggi (Treble):** Keluarkan suara bernada tinggi, dan saksikan bola melambung ke atas!
- **Nada Rendah (Bass):** Gunakan suara bernada rendah untuk mengarahkan bola turun dengan presisi.
- **Diam atau Nada Netral:** Jika Anda diam atau suara Anda berada dalam rentang netral, bola akan tetap diam pada posisinya saat itu, memungkinkan Anda untuk mengatur strategi.

**Tantangan & Tujuan:**
Navigasikan bola secara horizontal dari sisi kiri layar menuju sisi kanan untuk menyelesaikan setiap level. Namun, hati-hati! Terdapat dua penghalang, satu di atas dan satu di bawah, yang akan menguji kelincahan vokal Anda. Berhasil mencapai ujung kanan akan membawa Anda ke level berikutnya, di mana tantangan meningkat dengan menyempitnya jarak antara kedua penghalang (hingga maksimal 5 level).

**Sistem Skor:**
Setiap pergerakan bola yang berhasil Anda kontrol akan berkontribusi pada skor Anda. Semakin jauh bola bergerak dan semakin banyak level yang Anda taklukkan, semakin tinggi skor yang akan Anda raih. Teruslah bersuara agar anda mendapatkan skor!

Siapkah Anda menguji kemampuan vokal dan refleks Anda dalam "Suaraku Terbang"?

## Identitas Pengembang

1. Ikhsannudin Lathief - 122140137
   _Github: [latiiippp](https://github.com/latiiippp)_

2. Rustian Afencius Marbun - 122140155
   _Github: [122140155-rustian-afencius](https://github.com/122140155-rustian-afencius)_

3. Eden Wijaya - 122140187
   _Github: [EdenWijaya](https://github.com/EdenWijaya)_

## Logbook Mingguan

| Tanggal       | Deskripsi Aktivitas                                                                                                                  | Kontributor             |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------- |
| _1 Mei 2025_  | Membuat repositori _Suaraku Terbang_                                                                                                 | Lathief                 |
| _2 Mei 2025_  | - Mendeteksi kamera dari webcam laptop.<br>- Mengatasi mirror webcam.<br>- Refactoring _error handling webcam_ dan _exit condition_. | Lathief, Afencius, Eden |
| _10 Mei 2025_ | Menambahkan penghalang atas dan bawah.                                                                                               | Lathief                 |
| _11 Mei 2025_ | Menambahkan efek kacamata.                                                                                                           | Eden                    |
| _12 Mei 2025_ | Menambahkan bola.                                                                                                                    | Afencius                |
| _25 Mei 2025_ | - Deteksi suara.<br>- Logika pergerakan bola atas bawah.                                                                             | Eden, Afencius          |
| _26 Mei 2025_ | - Logika pergerakan bola ke kanan.<br>- Logika leveling.                                                                             | Lathief                 |
| _29 Mei 2025_ | - Improve GUI dan scoring system.<br>- Layar game over.                                                                              | Afencius, Eden          |
| _30 Mei 2025_ | - Fix kecepatan bola.<br>- Fix logika skor dan modularisasi kode.                                                                    | Eden, Lathief           |
| _31 Mei 2025_ | Update README.md                                                                                                                     | Lathief                 |

## Demonstrasi Program

[Demonstrasi Suaraku Terbang](https://youtu.be/gLfBfn93XxQ?si=p_Sxa1XPKhcVgiYn)

## Instruksi Instalasi

1.  **Clone repository ini:**

    ```bash
    git clone https://github.com/latiiippp/suaraku-terbang.git
    cd suaraku-terbang
    ```

2.  **Buat dan aktifkan virtual environment (direkomendasikan):**

    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

    atau jika menggunakan uv

    ```bash
    uv venv --python=python3.10
    # Windows
    source .venv/Scripts/activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependensi yang dibutuhkan:**

    ```bash
    pip install -r requirements.txt
    ```

    atau jika menggunakan uv

    ```bash
    uv pip install -r requirements.txt
    ```

## Instruksi Penggunaan Program

1.  **Jalankan game:**

    ```bash
    python main.py
    ```

2.  **Konfigurasi Audio (Jika di terminal terdapat error deteksi device mic):**
    Permainan ini menggunakan input mikrofon untuk mengontrol bola. Jika Anda mengalami masalah dengan deteksi suara, Anda mungkin perlu menyesuaikan pengaturan perangkat audio:

    - **Menemukan ID Perangkat Audio:**
      Skrip [`main.py`](main.py) akan mencetak ke terminal daftar perangkat audio yang tersedia saat dijalankan. Perhatikan ID perangkat input (mikrofon) yang ingin Anda gunakan (default ID 1).

      ```python
      // suaraku-terbang\main.py
      // ...existing code...
      # Cek perangkat audio yang tersedia (opsional, untuk debugging)
      print(sd.query_devices())
      // ...existing code...
      ```

    - **Mengubah ID Perangkat, Sample Rate, atau Channel di `audio_processing.py`:**
      Buka file [`audio_processing.py`](audio_processing.py).
      Cari baris berikut di dalam fungsi [`detect_sound_direction`](audio_processing.py):

      ```python
      // suaraku-terbang\audio_processing.py
      // ...existing code...
      recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=1) # Sesuaikan device jika perlu
      // ...existing code...
      ```

      Ganti nilai `device=1` dengan ID perangkat mikrofon Anda yang benar. Anda juga dapat menyesuaikan `sample_rate` atau `channels` jika diperlukan, meskipun nilai default biasanya sudah cukup baik.

      Jika ingin lebih memastikan nilai channel dan sampling rate dari mic yang anda gunakan, anda dapat melihatnya melalui settings bawaan windows.

      ![Settings/System/Sound/Properties](for_readme/format%20mic%20device.png)

3.  **Gameplay:**
    - Gunakan suara Anda untuk mengontrol gerakan bola. Suara bernada tinggi (treble) akan menggerakkan bola ke atas, dan suara bernada rendah (bass) akan menggerakkan bola ke bawah.
    - Hindari menabrak penghalang.
    - Capai ujung kanan layar untuk naik ke level berikutnya.
    - Terus gerakkan bola untuk mendapatkan skor lebih banyak.
    - Tekan tombol `Spasi` di layar awal untuk memulai permainan.
    - Tekan tombol `Esc` kapan saja untuk keluar dari permainan.
