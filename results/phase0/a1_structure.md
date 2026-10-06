# A1 Qasper structure confound (PILOT v3 data, exploratory)

Length-matched window: 203 cl100k tokens (median struct chunk), overlap 25.

| chunk set | chunks | median tok | greedy-oracle rec@512t | evidence not whole in 1 chunk | evidence <50% in any chunk |
|---|---|---|---|---|---|
| fixed512 | 1329 | 104 | 0.993 | 83.9% | 14.0% |
| fixedtok | 644 | 256 | 0.949 | 37.1% | 0.0% |
| fixedlen | 805 | 203 | 0.961 | 53.1% | 0.0% |
| parapack | 569 | 251 | 0.941 | 0.0% | 0.0% |
| struct | 715 | 203 | 0.968 | 0.0% | 0.0% |
| merged_rk | 621 | 221 | 0.951 | 0.0% | 0.0% |
| merged_nork | 679 | 209 | 0.962 | 0.0% | 0.0% |

rec@512t means (union evidence):

| system | bm25 | nomic | hybrid | mxbai |
|---|---|---|---|---|
| fixed512/text | 0.212 | 0.285 | 0.248 | 0.278 |
| fixedtok/text | 0.265 | 0.390 | 0.350 | 0.398 |
| fixedlen/text | 0.199 | 0.223 | 0.234 | 0.322 |
| parapack/text | 0.249 | 0.385 | 0.354 | 0.392 |
| struct/text | 0.311 | 0.494 | 0.478 | 0.419 |
| struct/tc | 0.324 | 0.489 | 0.470 | 0.419 |
| cr/cr | 0.367 | 0.478 | 0.488 | 0.440 |
| enr_rk/meta | 0.391 | 0.469 | 0.447 | 0.395 |
| enr_nork/meta | 0.416 | 0.491 | 0.489 | 0.413 |
| merged_rk/text | 0.375 | 0.505 | 0.458 | 0.434 |
| merged_rk/meta | 0.361 | 0.464 | 0.471 | 0.413 |
| merged_nork/meta | 0.421 | 0.472 | 0.502 | 0.426 |

(c) paired differences rec@512t (points):

| A - B | retr | diff | 95% CI | p_perm | TOST±5 p |
|---|---|---|---|---|---|
| parapack/text - fixedtok/text | bm25 | -1.6 | [-8.4, +5.7] | 0.695 | 0.185 |
| parapack/text - fixedtok/text | nomic | -0.5 | [-7.6, +7.6] | 0.919 | 0.120 |
| parapack/text - fixedtok/text | hybrid | +0.3 | [-11.1, +11.4] | 0.969 | 0.240 |
| parapack/text - fixed512/text | bm25 | +3.7 | [-6.6, +14.3] | 0.514 | 0.414 |
| parapack/text - fixed512/text | nomic | +10.0 | [+1.7, +18.5] | 0.026 | 0.874 |
| parapack/text - fixed512/text | hybrid | +10.6 | [-0.7, +22.2] | 0.089 | 0.816 |
| struct/text - parapack/text | bm25 | +6.2 | [-0.2, +12.7] | 0.093 | 0.618 |
| struct/text - parapack/text | nomic | +10.9 | [+3.8, +18.0] | 0.007 | 0.938 |
| struct/text - parapack/text | hybrid | +12.4 | [+5.1, +20.3] | 0.003 | 0.961 |
| struct/text - fixedlen/text | bm25 | +11.2 | [+4.8, +17.8] | 0.002 | 0.957 |
| struct/text - fixedlen/text | nomic | +27.1 | [+19.1, +36.1] | 0.000 | 1.000 |
| struct/text - fixedlen/text | hybrid | +24.4 | [+16.4, +33.7] | 0.000 | 1.000 |
| fixedlen/text - fixedtok/text | bm25 | -6.6 | [-14.0, +0.7] | 0.109 | 0.662 |
| fixedlen/text - fixedtok/text | nomic | -16.7 | [-24.9, -8.8] | 0.000 | 0.994 |
| fixedlen/text - fixedtok/text | hybrid | -11.7 | [-23.3, -1.7] | 0.039 | 0.866 |
| struct/text - fixedtok/text | bm25 | +4.6 | [-3.6, +12.7] | 0.310 | 0.456 |
| struct/text - fixedtok/text | nomic | +10.4 | [+3.1, +18.3] | 0.008 | 0.919 |
| struct/text - fixedtok/text | hybrid | +12.7 | [+5.5, +20.2] | 0.002 | 0.975 |

(d) union vs max-over-annotators (32/79 questions have >=2 distinct annotator evidence sets):

| A - B | retr | union diff [95% CI] | max diff [95% CI] |
|---|---|---|---|
| struct/text - fixedtok/text | bm25 | +4.6 [-3.6, +12.7] | +6.9 [-2.9, +16.6] |
| struct/text - fixedtok/text | nomic | +10.4 [+3.1, +18.3] | +15.4 [+7.8, +23.1] |
| struct/text - fixedtok/text | hybrid | +12.7 [+5.5, +20.2] | +15.4 [+7.8, +23.2] |
| struct/text - fixedtok/text | mxbai | +2.1 [-6.4, +10.9] | +4.6 [-4.9, +14.3] |
| struct/text - fixed512/text | bm25 | +9.9 [+0.6, +18.3] | +12.4 [+1.4, +22.3] |
| struct/text - fixed512/text | nomic | +20.9 [+12.5, +29.9] | +28.6 [+18.5, +38.9] |
| struct/text - fixed512/text | hybrid | +23.0 [+12.9, +33.5] | +27.6 [+15.8, +39.5] |
| struct/text - fixed512/text | mxbai | +14.1 [+4.9, +22.8] | +18.5 [+9.0, +28.0] |
| enr_rk/meta - struct/tc | bm25 | +6.8 [-0.8, +14.8] | +5.3 [-2.6, +13.3] |
| enr_rk/meta - struct/tc | nomic | -2.0 [-10.8, +7.4] | -3.8 [-13.6, +6.8] |
| enr_rk/meta - struct/tc | hybrid | -2.3 [-8.6, +4.3] | -4.2 [-11.7, +3.5] |
| enr_rk/meta - struct/tc | mxbai | -2.3 [-12.2, +7.1] | -2.4 [-12.4, +7.4] |
| enr_rk/meta - cr/cr | bm25 | +2.4 [-5.6, +10.4] | +1.7 [-6.2, +9.7] |
| enr_rk/meta - cr/cr | nomic | -0.9 [-7.9, +6.7] | +0.3 [-7.7, +9.6] |
| enr_rk/meta - cr/cr | hybrid | -4.1 [-10.4, +2.3] | -4.4 [-11.2, +2.5] |
| enr_rk/meta - cr/cr | mxbai | -4.5 [-12.7, +4.0] | -7.2 [-15.7, +1.8] |
| merged_rk/meta - merged_nork/meta | bm25 | -6.0 [-12.8, -0.2] | -7.6 [-14.8, -0.7] |
| merged_rk/meta - merged_nork/meta | nomic | -0.8 [-8.3, +7.1] | -1.3 [-9.4, +7.1] |
| merged_rk/meta - merged_nork/meta | hybrid | -3.1 [-10.0, +3.3] | -5.1 [-13.5, +2.4] |
| merged_rk/meta - merged_nork/meta | mxbai | -1.3 [-9.5, +6.7] | -3.2 [-12.0, +5.0] |
| merged_rk/meta - enr_rk/meta | bm25 | -3.1 [-9.3, +3.1] | -2.5 [-8.9, +3.6] |
| merged_rk/meta - enr_rk/meta | nomic | -0.5 [-4.3, +3.5] | -0.3 [-4.9, +4.1] |
| merged_rk/meta - enr_rk/meta | hybrid | +2.4 [-0.2, +5.9] | +3.2 [+0.6, +6.6] |
| merged_rk/meta - enr_rk/meta | mxbai | +1.8 [-3.6, +7.6] | +1.6 [-3.8, +7.4] |
