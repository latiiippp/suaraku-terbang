# suaraku-terbang

## Deskripsi

"Suaraku Terbang" merupakan sebuah filter tiktok yang dapat menangkap suara pengguna. Terdapat bola di permainan yang harus digerakkan oleh pengguna menggunakan suaranya. Filter ini mendeteksi suara nada tinggi (treble) dan nada rendah (bass) pengguna. Jika pengguna bersuara treble maka bola akan bergerak naik, jika bersuara bass maka bola bergerak turun, namun jika pengguna diam maka bola tidak bergerak. Terdapat 2 penghalang di atas dan bawah yang akan membatasi gerak bola. Pengguna harus terus menggerakan bola dari ujung kiri ke ujung kanan. Jika sudah berhasil mencapai ujung kanan, maka pengguna akan masuk ke level berikutnya dengan jarak antara kedua penghalang yang semakin sempit (maks level 5). Semakin banyak bola bergerak, maka skor yang diraih oleh pengguna semakin banyak.

## Identitas Pengembang

1. Ikhsannudin Lathief - 122140137
   _Github: [latiiippp](https://github.com/latiiippp)_

2. Rustian Afencius Marbun - 122140155
   _Github: [122140155-rustian-afencius](https://github.com/122140155-rustian-afencius)_

3. Eden Wijaya - 122140187
   _Github: [EdenWijaya](https://github.com/EdenWijaya)_

## Logbook Mingguan

| Tanggal       | Deskripsi Aktivitas                                                                                                                  | Kontributor |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
| _1 Mei 2025_  | Membuat repositori _Suaraku Terbang_                                                                                                 |             |
| _2 Mei 2025_  | - Mendeteksi kamera dari webcam laptop.<br>- Mengatasi mirror webcam.<br>- Refactoring _error handling webcam_ dan _exit condition_. |             |
| _10 Mei 2025_ | - Update README.md<br>- Membuat penghalang atas dan bawah.                                                                           |             |
|               |                                                                                                                                      |             |
|               |                                                                                                                                      |             |

## Instruksi Instalasi

1.  **Clone repository ini:**

    ```bash
    git clone https://github.com/username/suaraku-terbang.git
    cd suaraku-terbang
    ```

    _(Ganti `username` dengan nama pengguna GitHub pemilik repositori jika berbeda)_

2.  **Buat dan aktifkan virtual environment (direkomendasikan):**

    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependensi yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

## Instruksi Penggunaan Program

1.  **Jalankan game:**

    ```bash
    python main.py
    ```

2.  **Konfigurasi Audio (Jika Diperlukan):**
    Permainan ini menggunakan input mikrofon untuk mengontrol bola. Jika Anda mengalami masalah dengan deteksi suara, Anda mungkin perlu menyesuaikan pengaturan perangkat audio:

    - **Menemukan ID Perangkat Audio:**
      Skrip [`main.py`](d%3A%5CKULIAH%5CSEMESTER%206%5CMultimedia%5Ctubes%5Csuaraku-terbang%5Cmain.py) akan mencetak daftar perangkat audio yang tersedia saat dijalankan. Perhatikan ID perangkat input (mikrofon) yang ingin Anda gunakan.

      ```python
      // filepath: d:\KULIAH\SEMESTER 6\Multimedia\tubes\suaraku-terbang\main.py
      // ...existing code...
      # Cek perangkat audio yang tersedia (opsional, untuk debugging)
      print(sd.query_devices())
      // ...existing code...
      ```

    - **Mengubah ID Perangkat, Sample Rate, atau Channel di `audio_processing.py`:**
      Buka file [`audio_processing.py`](d%3A%5CKULIAH%5CSEMESTER%206%5CMultimedia%5Ctubes%5Csuaraku-terbang%5Caudio_processing.py).
      Cari baris berikut di dalam fungsi [`detect_sound_direction`](d%3A%5CKULIAH%5CSEMESTER%206%5CMultimedia%5Ctubes%5Csuaraku-terbang%5Caudio_processing.py):
      ```python
      // filepath: d:\KULIAH\SEMESTER 6\Multimedia\tubes\suaraku-terbang\audio_processing.py
      // ...existing code...
      recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=31) # Sesuaikan device jika perlu
      // ...existing code...
      ```
      Ganti nilai `device=31` dengan ID perangkat mikrofon Anda yang benar. Anda juga dapat menyesuaikan `sample_rate` atau `channels` jika diperlukan, meskipun nilai default biasanya sudah cukup baik.

3.  **Gameplay:**
    - Gunakan suara Anda untuk mengontrol bola. Suara bernada tinggi akan menggerakkan bola ke atas, dan suara bernada rendah akan menggerakkan bola ke bawah.
    - Hindari menabrak penghalang.
    - Capai ujung kanan layar untuk naik ke level berikutnya.
    - Tekan tombol `Spasi` di layar awal untuk memulai permainan.
    - Tekan tombol `Esc` kapan saja untuk keluar dari permainan.

````// filepath: d:\KULIAH\SEMESTER 6\Multimedia\tubes\suaraku-terbang\README.md
# suaraku-terbang

## Deskripsi

"Suaraku Terbang" adalah permainan interaktif di mana pemain mengontrol bola menggunakan suara mereka. Nada tinggi akan menggerakkan bola ke atas, dan nada rendah akan menggerakkan bola ke bawah. Tujuan permainan ini adalah untuk melewati rintangan dan mencapai skor setinggi mungkin. Permainan ini juga dilengkapi dengan deteksi wajah untuk menampilkan overlay kacamata pada pemain secara real-time.

## Identitas Pengembang

1. Ikhsannudin Lathief - 122140137
   _Github: [latiiippp](https://github.com/latiiippp)_

2. Rustian Afencius Marbun - 122140155
   _Github: [122140155-rustian-afencius](https://github.com/122140155-rustian-afencius)_

3. Eden Wijaya - 122140187
   _Github: [EdenWijaya](https://github.com/EdenWijaya)_

## Logbook Mingguan

| Tanggal       | Deskripsi Aktivitas                                                                                                                               | Kontributor        |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| _1 Mei 2025_  | Membuat repositori _Suaraku Terbang_                                                                                                              |                    |
| _2 Mei 2025_  | - Mendeteksi kamera dari webcam laptop.<br>- Mengatasi mirror webcam.<br>- Refactoring _error handling webcam_ dan _exit condition_.                 |                    |
| _10 Mei 2025_ | - Update README.md<br>- Membuat penghalang atas dan bawah.                                                                                        |                    |
|               |                                                                                                                                                   |                    |
|               |                                                                                                                                                   |                    |

## Instruksi Instalasi

1.  **Clone repository ini:**
    ```bash
    git clone https://github.com/username/suaraku-terbang.git
    cd suaraku-terbang
    ```
    *(Ganti `username` dengan nama pengguna GitHub pemilik repositori jika berbeda)*

2.  **Buat dan aktifkan virtual environment (direkomendasikan):**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependensi yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

## Instruksi Penggunaan Program

1.  **Jalankan game:**
    ```bash
    python main.py
    ```

2.  **Konfigurasi Audio (Jika Diperlukan):**
    Permainan ini menggunakan input mikrofon untuk mengontrol bola. Jika Anda mengalami masalah dengan deteksi suara, Anda mungkin perlu menyesuaikan pengaturan perangkat audio:

    *   **Menemukan ID Perangkat Audio:**
        Skrip [`main.py`](d%3A%5CKULIAH%5CSEMESTER%206%5CMultimedia%5Ctubes%5Csuaraku-terbang%5Cmain.py) akan mencetak daftar perangkat audio yang tersedia saat dijalankan. Perhatikan ID perangkat input (mikrofon) yang ingin Anda gunakan.
        ```python
        // filepath: d:\KULIAH\SEMESTER 6\Multimedia\tubes\suaraku-terbang\main.py
        // ...existing code...
        # Cek perangkat audio yang tersedia (opsional, untuk debugging)
        print(sd.query_devices())
        // ...existing code...
        ```

    *   **Mengubah ID Perangkat, Sample Rate, atau Channel di `audio_processing.py`:**
        Buka file [`audio_processing.py`](d%3A%5CKULIAH%5CSEMESTER%206%5CMultimedia%5Ctubes%5Csuaraku-terbang%5Caudio_processing.py).
        Cari baris berikut di dalam fungsi [`detect_sound_direction`](d%3A%5CKULIAH%5CSEMESTER%206%5CMultimedia%5Ctubes%5Csuaraku-terbang%5Caudio_processing.py):
        ```python
        // filepath: d:\KULIAH\SEMESTER 6\Multimedia\tubes\suaraku-terbang\audio_processing.py
        // ...existing code...
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=31) # Sesuaikan device jika perlu
        // ...existing code...
        ```
        Ganti nilai `device=31` dengan ID perangkat mikrofon Anda yang benar. Anda juga dapat menyesuaikan `sample_rate` atau `channels` jika diperlukan, meskipun nilai default biasanya sudah cukup baik.

3.  **Gameplay:**
    *   Tekan tombol `Spasi` di layar awal untuk memulai permainan.
    *   Tekan tombol `Esc` kapan saja untuk keluar dari permainan.
    *   Gunakan suara Anda untuk mengontrol bola. Suara bernada tinggi akan menggerakkan bola ke atas, dan suara bernada rendah akan menggerakkan bola ke bawah.
    *   Hindari menabrak penghalang.
    *   Capai ujung kanan layar untuk naik ke level berikutnya.
````
