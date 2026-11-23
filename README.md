# Studi Komparatif Empiris Model Generatif: AE, Sparse AE, VAE, GAN, DCGAN, DDPM

Tugas Take-Home Deep Learning (D6421) – Sesi 07: Generative Models

|               |                                                         |
| ------------- | ------------------------------------------------------- |
| **Nama**      | `Derry Riccardo`                                        |
| **NIM**       | `2802399513`                                            |
| **Kelas**     | `Deep Learning LB01`                                    |
| **Dataset**   | Fashion-MNIST (28×28 grayscale, 10 kelas, 70.000 citra) |
| **Framework** | PyTorch (Jupyter Notebook)                              |

---

## 1. Struktur Folder

```
Tugas AE dan GAN/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                      # semua kode (Jupyter Notebook)
│   ├── 00_helpers.ipynb
│   ├── 01_mlp_classifier.ipynb
│   ├── 02_vanilla_autoencoder.ipynb
│   ├── 03_sparse_autoencoder.ipynb
│   ├── 04_vae.ipynb
│   ├── 05_vanilla_gan.ipynb
│   ├── 06_dcgan.ipynb
│   ├── 07_ddpm.ipynb
│   ├── 08_eksperimen_batasan_mlp.ipynb
│   └── 09_evaluasi_tabel_dan_figur.ipynb
├── data/                     # dibuat otomatis: dataset Fashion-MNIST (tidak di-commit)
├── checkpoints/              # dibuat otomatis: bobot model terlatih (tidak di-commit)
└── results/                  # dibuat otomatis: semua hasil untuk laporan
    ├── tables/               #   tabel metrik, tabel konfigurasi, angka pendukung
    ├── figures/              #   Figur 1–5 + figur pendukung S1–S4 (PNG 300 dpi + PDF)
    ├── metrics/              #   angka lengkap per model (.json)
    ├── histories/            #   riwayat loss per epoch/iterasi (.json)
    ├── artifacts/            #   data gambar kecil untuk menyusun figur (.pt)
    └── split_info.json       #   jumlah data train / val / test
```

Folder `data/`, `checkpoints/`, dan `results/` **dibuat otomatis** saat notebook dijalankan. Lokasinya selalu di folder utama proyek, walaupun notebook-nya ada di dalam `src/`.

## 2. Isi Notebook (`src/`)

| Notebook                            | Isi                                                                                                                                                                                  | Bagian laporan        |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------- |
| `00_helpers.ipynb`                  | Fungsi bersama: konfigurasi, seed 42, split 80/10/10, metrik (MSE, PSNR, SSIM, FID, latency). **Tidak dijalankan sendiri**; dipanggil notebook lain lewat `%run ./00_helpers.ipynb`. | Standardisasi         |
| `01_mlp_classifier.ipynb`           | **Tahap 1:** MLP Classifier 784→256→64→10 sampai test acc ≥ 88%. Menyimpan W1, W2, W3 dan z_MLP.                                                                                     | III                   |
| `02_vanilla_autoencoder.ipynb`      | Vanilla AE                                                                                                                                                                           | II, IV, Figur 1, 4    |
| `03_sparse_autoencoder.ipynb`       | Sparse AE (+ uji pengaruh λ)                                                                                                                                                         | II, IV, Figur 1, 4    |
| `04_vae.ipynb`                      | Variational Autoencoder                                                                                                                                                              | II, IV, Figur 1, 2, 4 |
| `05_vanilla_gan.ipynb`              | Vanilla GAN berbasis MLP                                                                                                                                                             | II, IV, Figur 2, 4    |
| `06_dcgan.ipynb`                    | Deep Convolutional GAN                                                                                                                                                               | II, IV, Figur 2, 4    |
| `07_ddpm.ipynb`                     | Denoising Diffusion Probabilistic Model                                                                                                                                              | II, IV, Figur 2, 5    |
| `08_eksperimen_batasan_mlp.ipynb`   | **Tahap 2–5:** bukti bahwa bobot dan z_MLP tidak bisa dipakai untuk membuat gambar                                                                                                   | III, Figur 3          |
| `09_evaluasi_tabel_dan_figur.ipynb` | Mengumpulkan semua hasil → tabel metrik + Figur 1, 2, 4, 5 + figur pendukung S1–S4                                                                                                   | II, IV, V             |

Setiap notebook model memiliki susunan yang sama:
**Load Dataset → Hyperparameter → Model → Loss Function → Training → Plot Loss → Evaluasi Test Set → Analisis Tambahan → Simpan Hasil.**
Di bagian atas setiap notebook ada penjelasan singkat **hubungan loss function → hasil visual** model tersebut.

## 3. Instalasi

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

Sangat disarankan memakai **GPU** (CUDA) atau **Google Colab**. Untuk PyTorch versi CUDA, ikuti perintah di https://pytorch.org/get-started/locally/.

**Kalau memakai Google Colab:** unggah seluruh folder proyek (termasuk `requirements.txt` dan folder `src/`) ke Google Drive, lalu pastikan folder kerja notebook adalah folder `src/` (misalnya `%cd /content/drive/MyDrive/Tugas AE dan GAN/src`). Ini perlu karena setiap notebook memanggil `%run ./00_helpers.ipynb`, dan hasil disimpan di folder utama proyek (folder yang berisi `requirements.txt`).

Dataset Fashion-MNIST otomatis diunduh oleh `torchvision.datasets.FashionMNIST` ke folder `data/`. Bobot InceptionV3 untuk FID otomatis diunduh oleh `pytorch-fid`.

## 4. Cara Menjalankan

**Langkah 1 – tes cepat (disarankan).** Buka `src/00_helpers.ipynb`, ubah `QUICK = False` menjadi `QUICK = True`, simpan, lalu jalankan notebook 01 → 09 secara berurutan (_Run All_). Mode ini memakai 1 epoch dan data kecil, hanya untuk memastikan tidak ada error. Hasilnya disimpan terpisah di `results_quick/`.

**Langkah 2 – eksperimen penuh.** Kembalikan ke `QUICK = False`, lalu jalankan lagi **berurutan**:

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

Urutan ini penting karena:

- notebook 01 (MLP Classifier) dipakai notebook 02–08 untuk menghitung _mode coverage_ dan memilih titik interpolasi GAN,
- notebook 08 butuh hasil notebook 01 dan 02,
- notebook 09 mengumpulkan hasil semua notebook.

Notebook 02–07 tidak saling bergantung, jadi boleh dijalankan dalam urutan apa saja setelah 01.

## 5. Standardisasi Evaluasi

- **Dataset sama** untuk semua model: Fashion-MNIST.
- **Split 80/10/10**: train dan test bawaan digabung (70.000 citra), diacak dengan `torch.Generator().manual_seed(42)`, lalu dibagi 56.000 train / 7.000 val / 7.000 test.
- **Seed tetap**: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, `cudnn.deterministic = True`.
- **DataLoader**: `batch_size = 128`; `shuffle=True` untuk train, `shuffle=False` untuk val/test.
- **Semua metrik numerik dihitung pada Test Set.**
- **Encoder Autoencoder dan MLP Classifier memiliki arsitektur yang sama** (784→256→64, ReLU). Jadi di Tahap 4, perbedaan z_AE dan z_MLP murni berasal dari **loss function** (MSE vs Cross-Entropy).

## 6. Output (folder `results/`)

| File                               | Untuk                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| `tables/metrics_table.md` / `.csv` | Tabel komparasi metrik (Bagian IV)                                                   |
| `tables/config_table.md`           | Tabel konfigurasi parameter (Bagian II)                                              |
| `tables/supporting_metrics.md`     | Ringkasan angka pendukung Q1–Q6 (mode coverage, sparsity, anomaly AUROC, pengaruh λ) |
| `metrics/*.json`                   | Semua angka per model, termasuk ringkasan eksperimen MLP (`mlp_experiment.json`)     |
| `tables/interpolation_metrics.csv` | Deteksi _latent voids_ sepanjang interpolasi (Q4)                                    |
| `figures/figure1–5_*.png`          | 5 figur wajib (300 dpi, juga tersedia dalam PDF)                                     |
| `figures/figureS1–S4_*.png`        | Figur pendukung analisis                                                             |
| `histories/*.json`                 | Riwayat loss setiap model                                                            |

## 7. Bukti Tambahan untuk Pertanyaan Penelitian

| Pertanyaan | Bukti                                                                                                | Lokasi                                 |
| ---------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Q1         | Pengaruh λ terhadap MSE dan jumlah neuron aktif                                                      | notebook 03 bagian 8f                  |
| Q2         | Filter W1 & proporsi neuron aktif (Vanilla vs Sparse AE)                                             | notebook 02/03 bagian 8a, Figur S1     |
| Q3         | D(x), D(G(z)), norma gradien loss _saturating_ vs _non-saturating_, snapshot mode collapse           | notebook 05/06 bagian 6, Figur S2      |
| Q1, Q3     | _Mode coverage_: sebaran kelas sampel sintetis                                                       | Figur S4                               |
| Q4         | Jarak z(α) ke kode laten data latih (_latent voids_) + sampling acak di ruang laten AE               | notebook 02/03 bagian 8d–8e, Figur S3  |
| Q5         | Rank, nullspace, ‖WᵀW − I‖, kontrol pseudo-inverse, aktivasi yang dipotong ReLU, variasi intra-kelas | notebook 08                            |
| Q6         | AUROC deteksi cacat sintetis (AE vs Sparse AE); latency per 1 sampel                                 | notebook 02/03 bagian 8b, tabel metrik |

## 8. Catatan Metodologis

- **VAE**: rekonstruksi untuk metrik memakai μ (tanpa sampling ε). Sampling generatif memakai z ~ N(0, I).
- **Rentang piksel**: data disimpan di [0, 1]. GAN, DCGAN, dan DDPM dilatih di [−1, 1] (output Tanh), lalu dikembalikan ke [0, 1] sebelum dihitung metrik.
- **FID**: InceptionV3 standar (`pytorch-fid`, pool3 2048-d). Citra 1 kanal diulang jadi 3 kanal dan diperbesar ke 299×299. Statistik citra asli dihitung dari seluruh Test Set. Jumlah sampel sintetis sama untuk semua model (`FID_NUM_SAMPLES`, default 5.000).
- **Latency**: rata-rata beberapa kali eksekusi setelah _warm-up_. AE = encode + decode; model generatif = sintesis dari prior; DDPM = seluruh T = 1000 langkah.
- **DDPM** memakai bobot EMA (decay 0,999) untuk sampling.
- **Label "Kestabilan Training"** bersifat heuristik (koefisien variasi loss di paruh akhir training). Cek juga Figur 5 dan Figur S2.

## 9. Referensi

- Kingma & Welling (2014). _Auto-Encoding Variational Bayes_.
- Goodfellow et al. (2014). _Generative Adversarial Nets_.
- Radford, Metz & Chintala (2016). _Unsupervised Representation Learning with Deep Convolutional GANs_.
- Ho, Jain & Abbeel (2020). _Denoising Diffusion Probabilistic Models_.
- Heusel et al. (2017). _GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium_ (FID).
- Xiao, Rasul & Vollgraf (2017). _Fashion-MNIST_.
