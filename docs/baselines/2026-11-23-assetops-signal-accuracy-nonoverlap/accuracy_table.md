| Signal | Model | dtype | Windows (scaled) | Filled history | MAE | MASE | SQL | MASE ratio of means | MASE geo-mean | Naive MASE | Naive SQL | History-mean MASE | max \|vTSFM−native\| SQL |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chiller_6/condenser_water_flow | timesfm25 | fp32 | 24 (24), 19 off-state | 0.1% | 922.917 | 24.211 | 23.807 | 0.638 | 0.169 | 23.182 | 23.182 | 23.185 | 0.000012 (parity failed) |
| chiller_6/condenser_water_flow | toto2 | fp32 | 24 (24), 19 off-state | 0.1% | 798.290 | 22.944 | 23.067 | 0.552 | 0.135 | 23.182 | 23.182 | 23.185 | 0.000005 (parity failed) |
| chiller_6/condenser_water_flow | moirai2 | fp32 | 24 (24), 19 off-state | 0.1% | 724.445 | 22.699 | 22.686 | 0.501 | 0.170 | 23.182 | 23.182 | 23.185 | 4.69e-07 (parity failed) |
| chiller_6/supply_temperature (unforecastable) | timesfm25 | fp32 | 24 (24) | 0.2% | 32.716 | 0.805 | 0.595 | 0.797 | 0.798 | 1.049 | 1.049 | 0.811 | 6.17e-08 |
| chiller_6/supply_temperature (unforecastable) | toto2 | fp32 | 24 (24) | 0.2% | 32.910 | 0.810 | 0.595 | 0.802 | 0.803 | 1.049 | 1.049 | 0.811 | 4.57e-08 |
| chiller_6/supply_temperature (unforecastable) | moirai2 | fp32 | 24 (24) | 0.2% | 32.959 | 0.809 | 0.603 | 0.803 | 0.804 | 1.049 | 1.049 | 0.811 | 2.93e-08 |
| hydraulic_pump_1/ps1_pressure | timesfm25 | fp32 | 17 (17) | 0.0% | 1.521 | 7.082 | 5.728 | 1.511 | 2.008 | 7.838 | 7.838 | 17.078 | 0.000008 |
| hydraulic_pump_1/ps1_pressure | toto2 | fp32 | 17 (17) | 0.0% | 1.888 | 4.669 | 3.991 | 1.876 | 1.820 | 7.838 | 7.838 | 17.078 | 0.000003 |
| hydraulic_pump_1/ps1_pressure | moirai2 | fp32 | 17 (17) | 0.0% | 1.510 | 8.155 | 6.678 | 1.500 | 2.361 | 7.838 | 7.838 | 17.078 | 0.000043 |
| metro_pump_1/motor_current | timesfm25 | fp32 | 11 (11) | 11.0% | 1.375 | 1.314 | 1.035 | 1.346 | 1.183 | 1.388 | 1.388 | 0.934 | 6.29e-07 |
| metro_pump_1/motor_current | toto2 | fp32 | 11 (11) | 11.0% | 1.419 | 1.463 | 1.069 | 1.389 | 1.288 | 1.388 | 1.388 | 0.934 | 5.36e-07 |
| metro_pump_1/motor_current | moirai2 | fp32 | 11 (11) | 11.0% | 1.368 | 1.371 | 1.145 | 1.339 | 1.132 | 1.388 | 1.388 | 0.934 | 1.79e-07 |
| metro_pump_1/oil_temperature | timesfm25 | fp32 | 11 (11) | 11.0% | 3.200 | 0.889 | 0.699 | 0.905 | 0.797 | 0.893 | 0.893 | 1.074 | 7.87e-07 |
| metro_pump_1/oil_temperature | toto2 | fp32 | 11 (11) | 11.0% | 3.271 | 0.924 | 0.731 | 0.926 | 0.816 | 0.893 | 0.893 | 1.074 | 4.43e-07 |
| metro_pump_1/oil_temperature | moirai2 | fp32 | 11 (11) | 11.0% | 3.372 | 0.919 | 0.715 | 0.954 | 0.810 | 0.893 | 0.893 | 1.074 | 4.37e-07 |

`chiller_6/supply_temperature` is unforecastable on this data: noise-like (lag-1 autocorrelation -0.01 to 0.12); the history mean scores within the models' range. It is scored but excluded from ranking claims.

**Paired comparison** (MASE; 95% percentile intervals from 2000 paired bootstrap resamples of windows, seed 0). Windows are resampled independently, but their histories overlap heavily, so the intervals are approximate and probably too narrow. Off-state is defined from the future labels, so the subset excluding it is a post hoc diagnostic. Rank 1 is the lowest median MAE within a window.

| Signal | dtype | Subset | Windows | Competitor | Ratio of means | Geo-mean | Mean rank | P(lowest) | ÷ lowest | Claim |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| chiller_6/condenser_water_flow | fp32 | all | 24 | timesfm25 | 0.638 [0.341, 0.981] | 0.169 [0.046, 0.641] | 2.58 | 0.12 | 1.27 [0.96, 2.02] | not distinguishable from lowest |
| chiller_6/condenser_water_flow | fp32 | all | 24 | toto2 | 0.552 [0.301, 0.768] | 0.135 [0.038, 0.496] | 1.83 | 0.04 | 1.10 [0.97, 1.39] | not distinguishable from lowest |
| chiller_6/condenser_water_flow | fp32 | all | 24 | moirai2 | 0.501 [0.261, 0.700] | 0.170 [0.057, 0.544] | 2.92 | 0.84 | 1.00 [1.00, 1.00] | lowest; not distinguishable from timesfm25, toto2 |
| chiller_6/condenser_water_flow | fp32 | all | 24 | seasonal_naive | 0.729 [0.395, 1.012] | 0.178 [0.042, 0.737] | 3.54 | 0.00 | 1.46 [1.27, 1.84] | higher than lowest |
| chiller_6/condenser_water_flow | fp32 | all | 24 | mean_history | 0.898 [0.790, 1.046] | 1.109 [0.746, 2.073] | 4.12 | 0.00 | 1.79 [1.20, 3.53] | higher than lowest |
| chiller_6/condenser_water_flow | fp32 | excluding off-state (post hoc) | 5 | timesfm25 | 0.758 [0.726, 0.785] | 0.758 [0.726, 0.785] | 1.60 | 0.97 | 1.00 [1.00, 1.00] | lowest; every other competitor higher |
| chiller_6/condenser_water_flow | fp32 | excluding off-state (post hoc) | 5 | toto2 | 0.761 [0.728, 0.787] | 0.761 [0.727, 0.787] | 2.20 | 0.02 | 1.00 [1.00, 1.01] | higher than lowest |
| chiller_6/condenser_water_flow | fp32 | excluding off-state (post hoc) | 5 | moirai2 | 0.780 [0.755, 0.806] | 0.779 [0.755, 0.806] | 3.60 | 0.00 | 1.03 [1.01, 1.05] | higher than lowest |
| chiller_6/condenser_water_flow | fp32 | excluding off-state (post hoc) | 5 | seasonal_naive | 1.009 [0.953, 1.040] | 1.007 [0.952, 1.040] | 5.00 | 0.00 | 1.33 [1.31, 1.36] | higher than lowest |
| chiller_6/condenser_water_flow | fp32 | excluding off-state (post hoc) | 5 | mean_history | 0.763 [0.728, 0.791] | 0.762 [0.727, 0.791] | 2.60 | 8.00e-03 | 1.01 [1.00, 1.01] | higher than lowest |
| chiller_6/supply_temperature | fp32 | all | 24 | timesfm25 | 0.797 [0.760, 0.837] | 0.798 [0.758, 0.840] | 2.21 | 0.82 | 1.00 [1.00, 1.00] | no claim: unforecastable signal |
| chiller_6/supply_temperature | fp32 | all | 24 | toto2 | 0.802 [0.764, 0.843] | 0.803 [0.762, 0.846] | 2.79 | 1.00e-03 | 1.01 [1.00, 1.01] | no claim: unforecastable signal |
| chiller_6/supply_temperature | fp32 | all | 24 | moirai2 | 0.803 [0.769, 0.840] | 0.804 [0.766, 0.843] | 2.92 | 0.04 | 1.01 [1.00, 1.02] | no claim: unforecastable signal |
| chiller_6/supply_temperature | fp32 | all | 24 | seasonal_naive | 1.040 [0.988, 1.093] | 1.040 [0.988, 1.094] | 5.00 | 0.00 | 1.30 [1.25, 1.36] | no claim: unforecastable signal |
| chiller_6/supply_temperature | fp32 | all | 24 | mean_history | 0.802 [0.759, 0.847] | 0.803 [0.758, 0.850] | 2.08 | 0.14 | 1.01 [1.00, 1.02] | no claim: unforecastable signal |
| hydraulic_pump_1/ps1_pressure | fp32 | all | 17 | timesfm25 | 1.511 [0.388, 3.305] | 2.008 [0.749, 5.169] | 2.35 | 0.13 | 1.01 [0.89, 1.09] | not distinguishable from lowest |
| hydraulic_pump_1/ps1_pressure | fp32 | all | 17 | toto2 | 1.876 [0.423, 3.656] | 1.820 [0.786, 4.059] | 2.47 | 0.19 | 1.25 [0.72, 2.42] | not distinguishable from lowest |
| hydraulic_pump_1/ps1_pressure | fp32 | all | 17 | moirai2 | 1.500 [0.387, 3.344] | 2.361 [0.867, 5.965] | 2.65 | 0.30 | 1.00 [1.00, 1.00] | lowest; not distinguishable from timesfm25, toto2, seasonal_naive |
| hydraulic_pump_1/ps1_pressure | fp32 | all | 17 | seasonal_naive | 1.517 [0.339, 3.460] | 2.196 [0.796, 5.811] | 3.18 | 0.37 | 1.01 [0.79, 1.14] | not distinguishable from lowest |
| hydraulic_pump_1/ps1_pressure | fp32 | all | 17 | mean_history | 2.944 [2.180, 4.472] | 7.240 [3.741, 13.943] | 4.35 | 1.50e-03 | 1.96 [1.18, 5.73] | higher than lowest |
| metro_pump_1/motor_current | fp32 | all | 11 | timesfm25 | 1.346 [1.016, 1.645] | 1.183 [0.893, 1.536] | 3.00 | 0.00 | 1.44 [1.10, 1.98] | higher than lowest |
| metro_pump_1/motor_current | fp32 | all | 11 | toto2 | 1.389 [1.028, 1.777] | 1.288 [0.930, 1.721] | 3.55 | 0.00 | 1.49 [1.16, 2.08] | higher than lowest |
| metro_pump_1/motor_current | fp32 | all | 11 | moirai2 | 1.339 [0.919, 1.762] | 1.132 [0.757, 1.649] | 3.09 | 6.00e-03 | 1.43 [1.08, 2.02] | higher than lowest |
| metro_pump_1/motor_current | fp32 | all | 11 | seasonal_naive | 1.353 [1.036, 1.692] | 1.282 [1.006, 1.615] | 3.64 | 0.00 | 1.45 [1.13, 2.07] | higher than lowest |
| metro_pump_1/motor_current | fp32 | all | 11 | mean_history | 0.934 [0.637, 1.260] | 0.800 [0.589, 1.077] | 1.73 | 0.99 | 1.00 [1.00, 1.00] | lowest; every other competitor higher |
| metro_pump_1/oil_temperature | fp32 | all | 11 | timesfm25 | 0.905 [0.637, 1.231] | 0.797 [0.616, 1.054] | 2.91 | 0.29 | 1.00 [1.00, 1.00] | lowest; not distinguishable from toto2, moirai2, seasonal_naive, mean_history |
| metro_pump_1/oil_temperature | fp32 | all | 11 | toto2 | 0.926 [0.656, 1.217] | 0.816 [0.597, 1.095] | 3.09 | 0.13 | 1.02 [0.92, 1.16] | not distinguishable from lowest |
| metro_pump_1/oil_temperature | fp32 | all | 11 | moirai2 | 0.954 [0.665, 1.252] | 0.810 [0.595, 1.091] | 2.82 | 0.12 | 1.05 [0.91, 1.25] | not distinguishable from lowest |
| metro_pump_1/oil_temperature | fp32 | all | 11 | seasonal_naive | 0.918 [0.704, 1.199] | 0.832 [0.675, 1.038] | 3.64 | 0.27 | 1.01 [0.91, 1.16] | not distinguishable from lowest |
| metro_pump_1/oil_temperature | fp32 | all | 11 | mean_history | 1.018 [0.665, 1.500] | 0.849 [0.587, 1.248] | 2.55 | 0.19 | 1.12 [0.90, 1.50] | not distinguishable from lowest |

**Model spread** (largest ÷ smallest mean of per-window ratios on the sampled windows; see the paired comparison before reading it as a ranking)

| Signal | dtype | Lowest MASE | Highest MASE | Ratio | Lowest SQL | Highest SQL | Ratio |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| chiller_6/condenser_water_flow | fp32 | moirai2 22.699 | timesfm25 24.211 | 1.07 | moirai2 22.686 | timesfm25 23.807 | 1.05 |
| chiller_6/supply_temperature | fp32 | timesfm25 0.805 | toto2 0.810 | 1.01 | toto2 0.595 | moirai2 0.603 | 1.01 |
| hydraulic_pump_1/ps1_pressure | fp32 | toto2 4.669 | moirai2 8.155 | 1.75 | toto2 3.991 | moirai2 6.678 | 1.67 |
| metro_pump_1/motor_current | fp32 | timesfm25 1.314 | toto2 1.463 | 1.11 | timesfm25 1.035 | moirai2 1.145 | 1.11 |
| metro_pump_1/oil_temperature | fp32 | timesfm25 0.889 | toto2 0.924 | 1.04 | timesfm25 0.699 | toto2 0.731 | 1.05 |
