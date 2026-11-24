# wrmodes timings (2026-11-24T13:12:26)

Machine: Intel(R) Xeon(R) Processor @ 2.30GHz, 2 logical cores (2 physical), 7.8 GB RAM, Linux-6.18.44-fc-v819-x86_64-with-glibc2.39; Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1; BLAS openblas 0.3.34.106.0 (1 threads), openblas 0.3.31.dev (1 threads); wrmodes 1.0.0b1 (ccaf180404)

## C1

COMSOL: – (no log entry)

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| 90 | 23 (18–90 GHz) | 1.097 | 1.093 | 0.0456 dB | 0.0071 dB |

## C1b

COMSOL: – (no log entry)

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| 60 | 3 (30–90 GHz) | 3.077 | 3.171 | 0.0022 dB | 0.0014 dB |
| 120 | 3 (30–90 GHz) | 3.876 | 3.849 | 0.0048 dB | 0.0055 dB |

## C2

COMSOL: – (no log entry)

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| sech | 22 (18–60 GHz) | 0.137 | 0.128 | 0.0074 dB | – |
| constant, user mesh | 3 (18–60 GHz) | 0.095 | 0.082 | 0.0051 dB | – |

## C3a

COMSOL: – (no log entry)

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| h = 2.0 mm | 3 (8–12 GHz) | 0.003 | 0.003 | 0.0046 (\|S11\| 0.0234) dB | 0.0180 dB |
| h = 1.6 mm | 3 (8–12 GHz) | 0.004 | 0.003 | 0.0043 (\|S11\| 0.0197) dB | 0.0103 dB |

## C3b

COMSOL: – (no log entry)

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| 1.6 mm x 100 mm | 3 (10–60 GHz) | 0.048 | 0.047 | 2.0590 (\|S11\| 0.1106) dB | – |

## E1

| COMSOL run | DOF | solution time (s) | solves (f x parameters) | time per solve (s) |
|---|---|---|---|---|
| E1_miter_E_plain | 400982 | 81.0 | 14 | 5.79 |
| E1_miter_E_d087 | 410560 | 87.0 | 14 | 6.21 |
| E1_miter_E_d087_fine | 3374362 | 599.0 | 14 | 42.79 |
| E1_miter_E_plain_fine | 3280686 | 570.0 | 14 | 40.71 |

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| plain_FEM | 14 (8–36 GHz) | 0.055 | 0.054 | 0.0078 dB | 0.0300 allowed |
| plain_MM | 14 (8–36 GHz) | 0.026 | 0.026 | 0.0076 dB | 0.0300 allowed |
| d087_FEM | 14 (8–36 GHz) | 0.057 | 0.056 | 0.0188 dB | 0.0468 allowed |
| d087_MM | 14 (8–36 GHz) | 0.102 | 0.100 | 0.0184 dB | 0.0468 allowed |

## E2

| COMSOL run | DOF | solution time (s) | solves (f x parameters) | time per solve (s) |
|---|---|---|---|---|
| E2_taper_parameter_sweep | 27294/42460/104163/163797 | 9.0 | 4 | 2.25 |

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| L100 | 1 (15–15 GHz) | 0.033 | 0.032 | 0.0000 dB | 0.0205 allowed |
| L200 | 1 (15–15 GHz) | 0.038 | 0.038 | 0.0000 dB | 0.0205 allowed |

## E3

| COMSOL run | DOF | solution time (s) | solves (f x parameters) | time per solve (s) |
|---|---|---|---|---|
| E3_strangled_PEC | 2055738 | 59.0 | 2 | 29.50 |
| E3_strangled_PEC_fine | 3647588 | 105.0 | 2 | 52.50 |
| E3_strangled_aluminum | 2055738 | 69.0 | 2 | 34.50 |

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| PEC | 2 (18–40 GHz) | 0.701 | 0.755 | 0.0007 dB | 0.0206 allowed |
| aluminium | 2 (18–40 GHz) | 0.824 | 0.847 | 0.0033 dB | 0.0206 allowed |

## E4

| COMSOL run | DOF | solution time (s) | solves (f x parameters) | time per solve (s) |
|---|---|---|---|---|
| E4_twist_triangle | 1751434 | 116.0 | 4 | 29.00 |
| E4_twist_triangular_fine | 14356944 | 878.0 | 4 | 219.50 |
| E4_twist_cosine | 1748648 | 131.0 | 4 | 32.75 |

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| triangular | 4 (15–18 GHz) | 3.133 | 3.004 | 0.0003 (power) | 0.0020 allowed |
| cosine | 4 (15–18 GHz) | 3.058 | 3.187 | 0.0018 (power) | 0.0020 allowed |

## E5

| COMSOL run | DOF | solution time (s) | solves (f x parameters) | time per solve (s) |
|---|---|---|---|---|
| E5_two45_s2a_mesh_sweep | 212553/837382 | 18.0 | 4 | 4.50 |
| E5_two45_s3a_mesh_sweep | 244818/968850 | 21.0 | 4 | 5.25 |

| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \|Δ\| vs COMSOL finest (\|S21\|; steps also \|S11\|) | COMSOL mesh check / allowed |
|---|---|---|---|---|---|
| s2a_domain | 2 (30–56 GHz) | 1.616 | 1.525 | 0.0003 dB | 0.0224 allowed |
| s2a_cascade_cf1.5 | 2 (30–56 GHz) | 5.525 | 4.732 | 0.0012 dB | 0.0224 allowed |
| s2a_cascade_cf3 | 2 (30–56 GHz) | 111.143 | 104.434 | 0.0002 dB | 0.0224 allowed |
| s3a_domain | 2 (30–56 GHz) | 1.869 | 2.048 | 0.0008 dB | 0.0214 allowed |
| s3a_cascade_cf1.5 | 2 (30–56 GHz) | 4.790 | 5.347 | 0.0007 dB | 0.0214 allowed |
| s3a_cascade_cf3 | 2 (30–56 GHz) | 104.797 | 102.526 | 0.0000 dB | 0.0214 allowed |

