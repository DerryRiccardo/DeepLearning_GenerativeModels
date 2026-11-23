| Model | Lapisan Arsitektur | Dim Laten | Aktivasi | Loss | Optimizer | Learning Rate | Epoch |
|---|---|---|---|---|---|---|---|
| Vanilla AE | 784-256-64 / 64-256-784 | 64 | ReLU, Sigmoid (output) | MSE | Adam | 0.001 | 30 |
| Sparse AE | 784-256-64 / 64-256-784 | 64 (bottleneck) / 256 (sparse) | ReLU, Sigmoid (output) | MSE + lambda*||h||_1 (lambda = 3e-05) | Adam | 0.001 | 30 |
| VAE | 784-256-(mu, logvar 64) / 64-256-784 | 64 | ReLU, Sigmoid (output) | BCE + beta*KL (beta = 1.0) | Adam | 0.001 | 30 |
| Vanilla GAN | G: 100-256-512-1024-784 / D: 784-512-256-1 | 100 | LeakyReLU(0.2), Tanh (G output) | BCE adversarial (non-saturating) | Adam (b1 = 0.5, b2 = 0.999) | 0.0002 | 100 |
| DCGAN | G: ConvT 100->256x7x7->128x14x14->1x28x28 / D: Conv 1->64->128->1 | 100 | ReLU+BN (G), LeakyReLU(0.2)+BN (D), Tanh | BCE adversarial (non-saturating) | Adam (b1 = 0.5, b2 = 0.999) | 0.0002 | 30 |
| DDPM | U-Net 28->14->7 (ch 64/128/128), ResBlock+GroupNorm, attention 7x7, T = 1000 | 1x28x28 | SiLU | MSE prediksi noise (L_simple) | Adam + EMA 0.999 | 0.0002 | 40 |
| MLP Classifier | 784-256-64-10 | 64 (z_MLP) | ReLU, Softmax | Cross-Entropy | Adam | 0.001 | 23 (early stopping, target test acc >= 88%) |

Batch size 128, seed 42, split 80/10/10 (fashion_mnist).