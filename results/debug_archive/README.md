# Debug Archive

This folder contains faulty outputs and documentation of bugs that were fixed.

---

## Known Limitations

### Morphological Edge Artifacts (Not Fixed)
The `morphological` and `tophat` decomposition methods show edge artifacts where terrain features are cut off at the image boundary. This is expected behavior — the structuring element can't properly evaluate features that extend beyond the frame.

**Impact**: Minimal for real-world use. Map edges will be far from areas of interest in rendered images when using larger DEMs.

**Files**: `morphological_edge_fix_comparison.png`, `morphological_edge_mask.png` in `results/visualizations/`

---

## 2024-12-30: FFT Zero-Padding Bug

### Problem
The `fft_zeropad` upsampling method produced completely wrong output with vertical color bands instead of preserving terrain structure.

### Symptoms
- Output values ranged from -750 to +1000 instead of matching input (~808-1034 ft)
- Terrain structure was replaced with vertical stripes
- Caused horizontal banding artifacts in residual visualizations

### Root Cause
The original implementation used `fftshift`/`ifftshift` incorrectly when zero-padding the frequency spectrum. This scrambled the frequency components, especially with odd-dimension inputs (375 columns).

**Broken code:**
```python
fft = np.fft.fft2(dem_filled)
fft_shift = np.fft.fftshift(fft)

padded = np.zeros((new_h, new_w), dtype=complex)
pad_h = (new_h - h) // 2
pad_w = (new_w - w) // 2

padded[pad_h:pad_h + h, pad_w:pad_w + w] = fft_shift
result = np.real(np.fft.ifft2(np.fft.ifftshift(padded))) * (scale ** 2)
```

### Fix
Place frequency components in corners (standard FFT layout) rather than using fftshift, handling odd dimensions with ceiling division:

```python
fft = np.fft.fft2(dem_filled)
padded = np.zeros((new_h, new_w), dtype=complex)

h_half = (h + 1) // 2  # Ceiling division
w_half = (w + 1) // 2

# Place in corners
padded[:h_half, :w_half] = fft[:h_half, :w_half]
padded[:h_half, -(w - w_half):] = fft[:h_half, w_half:]
padded[-(h - h_half):, :w_half] = fft[h_half:, :w_half]
padded[-(h - h_half):, -(w - w_half):] = fft[h_half:, w_half:]

result = np.real(np.fft.ifft2(padded)) * (scale ** 2)
```

### Files
- `fft_broken_output.png` - The broken FFT upsampling result
- `fft_fixed_output.png` - The corrected result

