| Arsitektur Model | Dim Laten | Params | Training Time / Epoch (s) | Latency (ms / batch 64) | Latency (ms / 1 sampel) | Test MSE | Test PSNR (dB) | Test SSIM | FID Score | Kestabilan Training |
|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla Autoencoder | 64 | 435,536 | 1.50 | 0.35 | 0.35 | 0.00819 | 21.93 | 0.7417 | N/A (Latent Voids) | Stabil / Konvergen |
| Sparse Autoencoder | 64 (Bottleneck) / 256 (Sparse) | 435,536 | 1.23 | 0.30 | 0.30 | 0.00872 | 21.63 | 0.7337 | N/A (Latent Voids) | Stabil / Konvergen |
| Variational Autoencoder (VAE) | 64 | 451,984 | 1.52 | 0.18 | 0.16 | 0.01324 | 19.63 | 0.6465 | 87.86 | Stabil / Konvergen |
| Vanilla GAN (MLP) | 100 (Prior Noise) | 2,019,857 | 3.56 | 0.41 | 0.32 | N/A (Tanpa Encoder) | N/A | N/A | 106.81 | Cukup stabil |
| DCGAN | 100 (Prior Noise) | 1,920,194 | 6.63 | 1.82 | 0.50 | N/A (Tanpa Encoder) | N/A | N/A | 21.43 | Cukup stabil |
| DDPM | 1x28x28 (Dimensi Citra) | 4,659,777 | 73.91 | 21023.18 | 14283.75 | N/A (Denoising Model) | N/A | N/A | 9.54 | Stabil / Konvergen |
