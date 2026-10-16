# A6 v2-style statistics on v3 PILOT data

diff in points (x100); CIs: percentile cluster bootstrap 10k; p: cluster sign-flip 10k; TOST margin +-5 (Qasper rec@512t) / +-3 (FreshStack prec@1024t).
Holm families: {'v3_all_primary_cells': 22, 'v2_primary': 4, 'v2_rq1': 4}

| dataset | RQ | A - B | retr | diff | 95% CI | 90% CI | p_perm | Holm(v3 all) | Holm(v2 fam) | p_TOST | Holm TOST (v2 fam) | verdict (CI) | verdict (perm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qasper | RQ1 | struct/text - fixedtok/text | bm25 | +4.6 | [-3.6, +12.7] | [-2.4, +11.6] | 0.310 | 1.000 |  | 0.456 |  | inconclusive | inconclusive |
| qasper | RQ1 | struct/text - fixedtok/text | hybrid | +12.7 | [+5.5, +20.2] | [+6.7, +19.0] | 0.002 | 0.036 | 0.005 | 0.975 | 1.000 | different | different |
| qasper | RQ1 | struct/text - fixedtok/text | nomic | +10.4 | [+3.1, +18.3] | [+4.3, +17.1] | 0.008 |  |  | 0.919 |  | different | different |
| qasper | RQ1 | struct/text - fixedtok/text | mxbai | +2.1 | [-6.4, +10.9] | [-5.3, +9.4] | 0.671 |  |  | 0.269 |  | inconclusive | inconclusive |
| qasper | RQ1 | struct/text - fixed512/text | bm25 | +9.9 | [+0.6, +18.3] | [+2.1, +17.1] | 0.042 | 0.807 |  | 0.849 |  | different | different |
| qasper | RQ1 | struct/text - fixed512/text | hybrid | +23.0 | [+12.9, +33.5] | [+14.4, +31.7] | 0.000 | 0.002 | 0.000 | 0.999 | 1.000 | different | different |
| qasper | RQ1 | struct/text - fixed512/text | nomic | +20.9 | [+12.5, +29.9] | [+13.7, +28.3] | 0.000 |  |  | 1.000 |  | different | different |
| qasper | RQ1 | struct/text - fixed512/text | mxbai | +14.1 | [+4.9, +22.8] | [+6.4, +21.4] | 0.006 |  |  | 0.967 |  | different | different |
| qasper | RQ2 | enr_rk/meta - struct/tc | bm25 | +6.8 | [-0.8, +14.8] | [+0.2, +13.5] | 0.121 | 1.000 |  | 0.662 |  | inconclusive | inconclusive |
| qasper | RQ2 | enr_rk/meta - struct/tc | hybrid | -2.3 | [-8.6, +4.3] | [-7.6, +3.2] | 0.518 | 1.000 | 0.988 | 0.216 | 0.865 | inconclusive | inconclusive |
| qasper | RQ2 | enr_rk/meta - struct/tc | nomic | -2.0 | [-10.8, +7.4] | [-9.3, +5.8] | 0.675 |  |  | 0.264 |  | inconclusive | inconclusive |
| qasper | RQ2 | enr_rk/meta - struct/tc | mxbai | -2.3 | [-12.2, +7.1] | [-10.5, +5.5] | 0.641 |  |  | 0.297 |  | inconclusive | inconclusive |
| qasper | RQ3 | enr_rk/meta - cr/cr | bm25 | +2.4 | [-5.6, +10.4] | [-4.4, +9.2] | 0.601 | 1.000 |  | 0.280 |  | inconclusive | inconclusive |
| qasper | RQ3 | enr_rk/meta - cr/cr | hybrid | -4.1 | [-10.4, +2.3] | [-9.4, +1.2] | 0.247 | 1.000 | 0.988 | 0.401 | 0.865 | inconclusive | inconclusive |
| qasper | RQ3 | enr_rk/meta - cr/cr | nomic | -0.9 | [-7.9, +6.7] | [-6.8, +5.5] | 0.820 |  |  | 0.137 |  | inconclusive | inconclusive |
| qasper | RQ3 | enr_rk/meta - cr/cr | mxbai | -4.5 | [-12.7, +4.0] | [-11.4, +2.7] | 0.324 |  |  | 0.460 |  | inconclusive | inconclusive |
| qasper | RQ4b | merged_rk/meta - merged_nork/meta | bm25 | -6.0 | [-12.8, -0.2] | [-11.6, -1.0] | 0.092 | 1.000 |  | 0.607 |  | different | inconclusive |
| qasper | RQ4b | merged_rk/meta - merged_nork/meta | hybrid | -3.1 | [-10.0, +3.3] | [-8.8, +2.3] | 0.389 | 1.000 |  | 0.305 |  | inconclusive | inconclusive |
| qasper | RQ4b | merged_rk/meta - merged_nork/meta | nomic | -0.8 | [-8.3, +7.1] | [-7.1, +5.8] | 0.857 |  |  | 0.160 |  | inconclusive | inconclusive |
| qasper | RQ4b | merged_rk/meta - merged_nork/meta | mxbai | -1.3 | [-9.5, +6.7] | [-7.9, +5.4] | 0.822 |  |  | 0.195 |  | inconclusive | inconclusive |
| qasper | RQ4b | merged_rk/meta - enr_rk/meta | bm25 | -3.1 | [-9.3, +3.1] | [-8.3, +2.1] | 0.447 | 1.000 |  | 0.300 |  | inconclusive | inconclusive |
| qasper | RQ4b | merged_rk/meta - enr_rk/meta | hybrid | +2.4 | [-0.2, +5.9] | [+0.0, +5.3] | 0.252 | 1.000 |  | 0.065 |  | inconclusive | inconclusive |
| qasper | RQ4b | merged_rk/meta - enr_rk/meta | nomic | -0.5 | [-4.3, +3.5] | [-3.8, +2.8] | 0.842 |  |  | 0.015 |  | equivalent | equivalent |
| qasper | RQ4b | merged_rk/meta - enr_rk/meta | mxbai | +1.8 | [-3.6, +7.6] | [-2.7, +6.6] | 0.538 |  |  | 0.163 |  | inconclusive | inconclusive |
| freshstack | RQ1 | struct/text - fixedtok/text | bm25 | -2.2 | [-5.9, +1.5] | [-5.3, +0.9] | 0.255 | 1.000 |  | 0.336 |  | inconclusive | inconclusive |
| freshstack | RQ1 | struct/text - fixedtok/text | hybrid | -2.6 | [-6.4, +1.2] | [-5.8, +0.6] | 0.190 | 1.000 | 0.190 | 0.403 | 1.000 | inconclusive | inconclusive |
| freshstack | RQ1 | struct/text - fixedtok/text | nomic | -0.8 | [-4.9, +3.3] | [-4.3, +2.6] | 0.695 |  |  | 0.152 |  | inconclusive | inconclusive |
| freshstack | RQ1 | struct/text - fixedtok/text | mxbai | -1.0 | [-5.2, +3.3] | [-4.5, +2.7] | 0.658 |  |  | 0.179 |  | inconclusive | inconclusive |
| freshstack | RQ1 | struct/text - fixed512/text | bm25 | +3.0 | [+0.1, +5.9] | [+0.6, +5.5] | 0.050 | 0.902 |  | 0.497 |  | different | inconclusive |
| freshstack | RQ1 | struct/text - fixed512/text | hybrid | +5.1 | [+1.4, +8.9] | [+2.0, +8.3] | 0.010 | 0.196 | 0.020 | 0.864 | 1.000 | different | different |
| freshstack | RQ1 | struct/text - fixed512/text | nomic | +6.5 | [+2.9, +10.1] | [+3.5, +9.5] | 0.000 |  |  | 0.970 |  | different | different |
| freshstack | RQ1 | struct/text - fixed512/text | mxbai | +3.4 | [-0.1, +6.9] | [+0.4, +6.4] | 0.068 |  |  | 0.587 |  | inconclusive | inconclusive |
| freshstack | RQ2 | enr_rk/meta - struct/tc | bm25 | +1.0 | [-1.9, +3.8] | [-1.4, +3.3] | 0.509 | 1.000 |  | 0.082 |  | inconclusive | inconclusive |
| freshstack | RQ2 | enr_rk/meta - struct/tc | hybrid | +2.3 | [-1.5, +6.2] | [-0.9, +5.5] | 0.250 | 1.000 | 0.988 | 0.357 | 0.865 | inconclusive | inconclusive |
| freshstack | RQ2 | enr_rk/meta - struct/tc | nomic | +2.4 | [-2.0, +6.9] | [-1.3, +6.1] | 0.295 |  |  | 0.392 |  | inconclusive | inconclusive |
| freshstack | RQ2 | enr_rk/meta - struct/tc | mxbai | +4.6 | [+0.5, +8.6] | [+1.1, +8.0] | 0.032 |  |  | 0.775 |  | different | different |
| freshstack | RQ3 | enr_rk/meta - cr/cr | bm25 | +0.5 | [-2.4, +3.8] | [-2.0, +3.3] | 0.764 | 1.000 |  | 0.063 |  | inconclusive | inconclusive |
| freshstack | RQ3 | enr_rk/meta - cr/cr | hybrid | +1.7 | [-2.4, +5.9] | [-1.8, +5.3] | 0.445 | 1.000 | 0.988 | 0.272 | 0.865 | inconclusive | inconclusive |
| freshstack | RQ3 | enr_rk/meta - cr/cr | nomic | +0.9 | [-3.7, +5.5] | [-3.0, +4.7] | 0.711 |  |  | 0.185 |  | inconclusive | inconclusive |
| freshstack | RQ3 | enr_rk/meta - cr/cr | mxbai | +6.7 | [+3.1, +10.5] | [+3.6, +9.9] | 0.001 |  |  | 0.971 |  | different | different |
| freshstack | RQ4b | merged_rk/meta - enr_rk/meta | bm25 | -0.2 | [-2.6, +2.1] | [-2.2, +1.7] | 0.834 | 1.000 |  | 0.011 |  | equivalent | equivalent |
| freshstack | RQ4b | merged_rk/meta - enr_rk/meta | hybrid | -0.6 | [-2.5, +1.2] | [-2.2, +0.9] | 0.503 | 1.000 |  | 0.006 |  | equivalent | equivalent |
| freshstack | RQ4b | merged_rk/meta - enr_rk/meta | nomic | -0.5 | [-2.2, +1.1] | [-1.9, +0.9] | 0.609 |  |  | 0.001 |  | equivalent | equivalent |
| freshstack | RQ4b | merged_rk/meta - enr_rk/meta | mxbai | -0.7 | [-1.8, +0.4] | [-1.6, +0.3] | 0.249 |  |  | 0.000 |  | equivalent | equivalent |
