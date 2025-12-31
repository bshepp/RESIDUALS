# Differential Analysis Report

## Top Candidates for Archaeological Feature Detection

| Rank | Combination | Score | Linear Features | Spatial Autocorr | Feature SNR |
|------|-------------|-------|-----------------|------------------|-------------|
| 1 | polynomial_lanczos_vs_polynomial_fft_zeropad | 573.9 | 2 | 0.704 | 25.90 |
| 2 | polynomial_bicubic_vs_polynomial_fft_zeropad | 562.0 | 2 | 0.705 | 25.11 |
| 3 | polynomial_bspline_vs_polynomial_fft_zeropad | 561.8 | 2 | 0.705 | 25.10 |
| 4 | wavelet_dwt_lanczos_vs_morphological_lanczos | 435.7 | 20 | 0.986 | 5.56 |
| 5 | wavelet_dwt_bicubic_vs_morphological_bicubic | 435.6 | 20 | 0.986 | 5.51 |
| 6 | wavelet_dwt_bicubic_vs_morphological_bspline | 435.6 | 20 | 0.986 | 5.51 |
| 7 | wavelet_dwt_bspline_vs_morphological_bicubic | 435.5 | 20 | 0.987 | 5.51 |
| 8 | wavelet_dwt_bicubic_vs_morphological_lanczos | 435.2 | 20 | 0.986 | 5.50 |
| 9 | wavelet_dwt_bspline_vs_morphological_bspline | 435.2 | 20 | 0.987 | 5.51 |
| 10 | bilateral_fft_zeropad_vs_wavelet_dwt_fft_zeropad | 434.9 | 16 | 0.982 | 5.40 |
| 11 | wavelet_dwt_bspline_vs_morphological_lanczos | 434.8 | 20 | 0.986 | 5.50 |
| 12 | bilateral_bicubic_vs_tophat_lanczos | 429.6 | 20 | 0.997 | 6.30 |
| 13 | bilateral_bspline_vs_tophat_lanczos | 429.4 | 20 | 0.997 | 6.30 |
| 14 | bilateral_lanczos_vs_tophat_lanczos | 429.3 | 20 | 0.997 | 6.30 |
| 15 | bilateral_bicubic_vs_tophat_bicubic | 428.8 | 20 | 0.997 | 6.27 |
| 16 | bilateral_bicubic_vs_tophat_bspline | 428.8 | 20 | 0.997 | 6.27 |
| 17 | bilateral_bspline_vs_tophat_bicubic | 428.7 | 20 | 0.997 | 6.27 |
| 18 | bilateral_bspline_vs_tophat_bspline | 428.6 | 20 | 0.997 | 6.27 |
| 19 | morphological_lanczos_vs_tophat_bspline | 418.6 | 19 | 0.996 | 6.18 |
| 20 | wavelet_dwt_bicubic_vs_tophat_lanczos | 413.5 | 19 | 0.996 | 5.55 |

## Metric Explanations

- **Linear Features**: Count of detected linear structures (roads, walls)
- **Spatial Autocorr**: Moran's I (high = spatially structured)
- **Feature SNR**: Signal-to-noise ratio of prominent features
