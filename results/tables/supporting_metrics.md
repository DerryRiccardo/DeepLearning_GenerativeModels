**Mode coverage** (entropi 1.0 = semua kelas merata)

| Model | Entropi | Kelas tercakup |
|---|---|---|
| VAE | 0.985 | 10 / 10 |
| Vanilla GAN | 0.995 | 10 / 10 |
| DCGAN | 0.996 | 10 / 10 |
| DDPM | 0.994 | 10 / 10 |

**Sparsity & deteksi anomali**

| Model | Neuron aktif / gambar | Neuron mati | AUROC cacat | Rasio error cacat/normal |
|---|---|---|---|---|
| Vanilla AE | 52.7% | 86 | 0.7287 | 1.60 |
| Sparse AE | 19.1% | 150 | 0.7244 | 1.57 |
| VAE (referensi) | – | – | 0.6984 | 1.44 |

**Sampling acak di ruang laten AE** (alasan FID ditulis N/A (Latent Voids))

| Model | FID | Entropi kelas |
|---|---|---|
| Vanilla AE | 165.29 | 0.942 |
| Sparse AE | 173.02 | 0.924 |

**Pengaruh lambda pada Sparse AE** (training singkat, untuk Q1)

| lambda | Test MSE | Test SSIM | Neuron aktif / gambar |
|---|---|---|---|
| 0 | 0.01271 | 0.6553 | 52.1% |
| 1e-05 | 0.01247 | 0.6576 | 24.8% |
| 3e-05 | 0.01321 | 0.6404 | 14.2% |
| 0.0001 | 0.01793 | 0.5704 | 3.8% |
| 0.001 | 0.03006 | 0.4269 | 1.2% |