# RESIDUALS Rerun Plan

## Overview

This document outlines files that failed during the exhaustive run and redundancy analysis, and the plan to complete them.

## Failure Summary

### Exhaustive Run Failures (Original Generation)

| Category | Count | Cause |
|----------|-------|-------|
| Disk write failures | 1,030 | Windows: "X requested and 0 written" |
| Disk space failures | 2,366 | Unix: "No space left on device" |
| Decomposition failures | 3 | Code bugs (see below) |
| **Total** | **3,399** | |

### Redundancy Analysis Failures (Fingerprinting)

| Category | Count | Cause |
|----------|-------|-------|
| Memory failures | 1,030 | "Not enough memory resources" |

**Note**: The 1,030 memory failures in redundancy analysis are for files that DO exist but couldn't be loaded into RAM for fingerprinting (large upsampled files at scale 8/16).

## Decomposition Failures (Code Bugs)

These 3 failures indicate actual bugs that need fixing:

1. `dog_multiscale_base_sigma0.5_n_scales3_sigma_ratio1.4`: `name 'residual' is not defined`
   - **Action**: Review `dog_multiscale` implementation in `src/decomposition/methods_extended.py`

2. `rolling_ball_radius200` (failed 2x): Radius too large for DEM size
   - **Action**: This is an expected edge case (radius > DEM dimension). Document as known limitation.

## Disk Space Failures

These 3,396 files failed because D: drive ran out of space during the exhaustive run.

### Pattern Analysis

Failed files are predominantly:
- Large upsampling scales (scale8, scale16)
- Wavelet biorthogonal decompositions (levels 4-5)
- Late-alphabet decomposition methods (ran near end of exhaustive run)

### Files Affected

Major categories:
- `wavelet_biorthogonal_level4_*` combinations
- `wavelet_biorthogonal_level5_*` combinations  
- `wavelet_reverse_biorthogonal_*` combinations
- Various `*_scale8` and `*_scale16` upsampling

## Rerun Plan

### Phase 1: Fix Code Bugs

1. Review and fix `dog_multiscale` residual variable reference
2. Run quick test on fixed method
3. Commit fix

### Phase 2: Generate Missing Files

**Prerequisites**:
- Free up ~500GB on D: drive (or use different drive)
- Close other applications to maximize available RAM

**Commands**:
```bash
cd F:\science-projects\DIVERGE

# Extract list of failed files
python extract_failed_combinations.py --log exhaustive_experiment.log --output failed_combinations.txt

# Rerun only failed combinations
python run_exhaustive.py --output D:\DIVERGE_exhaustive --combinations-file failed_combinations.txt
```

### Phase 3: Re-run Redundancy Analysis on Large Files

The 1,030 files that couldn't be fingerprinted need special handling:

**Option A: Chunked Loading**
- Modify `analyze_redundancy.py` to use memory-mapped arrays
- Load files in chunks instead of all at once

**Option B: Reduced Fingerprint**
- Compute fingerprint on downsampled version of large files
- Less accurate but feasible with available RAM

**Option C: More RAM**
- Run on machine with 64GB+ RAM
- Or use cloud instance temporarily

### Phase 4: Verification

1. Regenerate checksums for new files
2. Rerun redundancy analysis
3. Verify no gaps in coverage

## Script: Extract Failed Combinations

Create `extract_failed_combinations.py`:

```python
#!/usr/bin/env python3
\"\"\"Extract failed combination IDs from exhaustive run log.\"\"\"

import re
import argparse
from pathlib import Path

def extract_failures(log_path: Path) -> list:
    failures = []
    pattern = r'Failed: ([^:]+):'
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'WARNING - Failed:' in line:
                match = re.search(pattern, line)
                if match:
                    failures.append(match.group(1).strip())
    
    return failures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    failures = extract_failures(Path(args.log))
    
    with open(args.output, 'w') as f:
        for combo in failures:
            f.write(combo + '\n')
    
    print(f"Extracted {len(failures)} failed combinations to {args.output}")

if __name__ == '__main__':
    main()
```

## Estimated Requirements

| Phase | Disk Space | RAM | Time |
|-------|-----------|-----|------|
| Phase 2 (generate) | ~500 GB | 16 GB | ~8 hours |
| Phase 3 (fingerprint) | 0 | 32-64 GB | ~2 hours |
| Phase 4 (verify) | 0 | 16 GB | ~1 hour |

## Status

- [ ] Phase 1: Fix code bugs
- [ ] Phase 2: Generate missing files
- [ ] Phase 3: Re-fingerprint large files
- [ ] Phase 4: Verification

---

*Generated: 2026-01-08*
