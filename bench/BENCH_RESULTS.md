# Bench results: 7 Kepler stars

Signal periods in days (4 decimals, sorted), one run on one machine. This is a check on 7 stars, not a quality estimate on a sample.

## Configurations

| Label | Outlier clipping | Alias resolution |
|---|---|---|
| A | symmetric | off (old search) |
| B | upward only | on (first variant of the branch) |
| C | upward only | off |
| D | symmetric | on (final) |

Configuration C was measured only on stars 11442793 and 6541920.

## Periods

| Star | Config | Periods, d |
|---|---|---|
| 4852528 (Kepler-80) | A, B, D | 0.9868, 3.0722, 7.0534, 9.5215 |
| 11442793 (Kepler-90) | A | 7.0084, 8.7200, 45.9687, 59.7367 |
| 11442793 (Kepler-90) | B, C | 59.7367, 70.2119, 82.8892, 97.8089 |
| 11442793 (Kepler-90) | D | 7.0084, 8.7200, 59.7367, 91.9373 (one period correction, factor 2.0) |
| 5780885 (Kepler-7) | A | 0.5066, 2.4427, 4.8854, 94.1801 |
| 5780885 (Kepler-7) | B | 0.5042, 4.8856 |
| 5780885 (Kepler-7) | D | 0.5066, 4.8854, 94.1801 |
| 6541920 (Kepler-11) | A, D | 10.3045, 13.0253, 22.6858, 31.9952 |
| 6541920 (Kepler-11) | B, C | 13.0253, 22.6858, 31.9952, 59.1764 |
| 8977179 (eclipsing binary) | A | 0.5323, 0.7988, 1.5973 |
| 8977179 (eclipsing binary) | B, D | 1.5973 |
| 10006641 | A | 0.8942, 0.8942 |
| 10006641 | B, D | 0.8942 |
| 10991769 (eclipsing binary) | A | 0.8202, 0.8203, 1.6409 |
| 10991769 (eclipsing binary) | B, D | 1.6409 |

## Signal counts

- A: 24 signals (14 MATCHED, 2 HARMONIC, 8 UNMATCHED).
- D: 18 signals (14 MATCHED, 0 HARMONIC, 4 UNMATCHED).

## Notes

- For the eclipsing binaries, a period equal to half the catalog period remains (3.196 and 3.281 d).
- For Kepler-90, 4 signals were found (limit set by `--max-signals`).
- On Kepler-80, results of A, B and D coincide.
