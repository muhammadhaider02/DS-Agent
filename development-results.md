# DS-Agent: Development Stage Results

This document consolidates all benchmark results from the **Development Stage** of DS-Agent across multiple model configurations. Each task is evaluated over **5 trials**. Failures (F) are assigned the worst rank during ranking.

## All Trial Results

Trial values are sorted from worst to best performance per metric direction.

| Task | Metric | DeepSeek-V3 | GPT-4 | Hybrid (R1+V3) |
|:-----|:-------|:------------|:------|:---------------|
| FB | MCRMSE ↓ | F, F, 0.5718, 0.5604, 0.5596 | 0.6352, 0.6036, 0.5971, 0.5384, 0.5277 | 0.5767, 0.5545, 0.5533, 0.5324, 0.5186 |
| AR | RMSE ↓ | F, 1.6702, 1.6544, 1.5346, 1.5345 | 2.7393, 2.4784, 1.8065, 1.6982, 1.5016 | 1.6550, 1.4397, 1.4277, 1.3719, 1.3135 |
| TE | Accuracy ↑ | F, 0.6658, 0.8178, 0.8427, 0.8490 | 0.2504, 0.3239, 0.6452, 0.6640, 0.7976 | 0.6661, 0.8213, 0.8272, 0.8529, 0.8529 |
| CP | Accuracy ↑ | 0.9436, 0.9761, 0.9761, 0.9778, 0.9810 | 0.9162, 0.9487, 0.9743, 0.9811, 0.9846 | 0.9692, 0.9795, 0.9812, 0.9857, 0.9857 |
| ETT | MSE ↓ | 0.5369, 0.4710, 0.4455, 0.3855, 0.3782 | 2.7290, 0.8566, 0.7363, 0.4071, 0.3222 | 1.1102, 1.0385, 0.9311, 0.6824, 0.5840 |
| ILI | MSE ↓ | 5.1587, 4.5930, 3.8230, 3.7345, 3.5825 | 5.5217, 4.6283, 4.5704, 4.2758, 3.5371 | 4.1982, 3.9816, 3.8244, 3.5232, 3.3182 |
| HW | Accuracy ↑ | 0.0365, 0.0953, 0.2306, 0.5396, 0.5494 | 0.0482, 0.0623, 0.0952, 0.1952, 0.2423 | 0.1059, 0.2859, 0.5294, 0.5400, 0.5553 |
| EC | Accuracy ↑ | 0.2624, 0.2624, 0.2776, 0.2814, 0.3307 | 0.2585, 0.2661, 0.3688, 0.3802, 0.6958 | 0.3308, 0.3346, 0.3346, 0.3346, 0.3422 |
| MCS | RMSLE ↓ | 0.3024, 0.2957, 0.2957, 0.2947, 0.2944 | 0.3020, 0.2974, 0.2970, 0.2916, 0.2858 | 0.3021, 0.2938, 0.2915, 0.2901, 0.2892 |
| WBY | MAE ↓ | F, 331.56, 331.29, 327.53, 313.63 | 325, 325, 325, 182, 182 | 310.59, 301.37, 288.83, 288.83, 276.33 |
| ST | Accuracy ↑ | 0.7929, 0.7962, 0.8011, 0.8011, 0.8024 | 0.7780, 0.7993, 0.8027, 0.8027, 0.8062 | 0.8022, 0.8045, 0.8063, 0.8068, 0.8085 |
| ES | AUROC ↑ | 0.7240, 0.7603, 0.7643, 0.9182, 0.9309 | 0.6258, 0.6258, 0.9576, 0.9761, 0.9761 | 0.7365, 0.7527, 0.7946, 0.8365, 0.9188 |

## Mean Rank per Task

All 15 runs (5 per config) ranked jointly per task. Lower is better.

| Model | FB | AR | TE | CP | ETT | ILI | HW | EC | MCS | WBY | ST | ES | **Avg** |
|:------|:--:|:--:|:--:|:--:|:---:|:---:|:--:|:--:|:---:|:---:|:--:|:--:|:-------:|
| **DeepSeek-V3** | 10.6 | 9.2 | 7.8 | 9.4 | 4.6 | 8.2 | 8.0 | 11.4 | 9.8 | 12.4 | 11.2 | 8.4 | **9.25** |
| **GPT-4** | 8.4 | 11.0 | 11.6 | 9.4 | 8.2 | 10.4 | 11.0 | 6.6 | 8.4 | 6.6 | 8.8 | 7.0 | **8.95** |
| **Hybrid (R1+V3)** | 5.0 | 3.8 | 4.6 | 5.2 | 11.2 | 5.4 | 5.0 | 6.0 | 5.8 | 5.0 | 4.0 | 8.6 | **5.80** |

## Best Rank per Task

Best rank = rank of the single best run out of 15 combined. Lower is better.

| Model | FB | AR | TE | CP | ETT | ILI | HW | EC | MCS | WBY | ST | ES | **Avg** |
|:------|:--:|:--:|:--:|:--:|:---:|:---:|:--:|:--:|:---:|:---:|:--:|:--:|:-------:|
| **DeepSeek-V3** | 7 | 6 | 3 | 6 | 2 | 4 | 2 | 9 | 7 | 8 | 8 | 4 | **5.50** |
| **GPT-4** | 2 | 5 | 8 | 3 | 1 | 3 | 7 | 1 | 1 | 1 | 4 | 1 | **3.08** |
| **Hybrid (R1+V3)** | 1 | 1 | 1 | 1 | 8 | 1 | 1 | 4 | 2 | 3 | 1 | 5 | **2.42** |

## Token Usage: Hybrid (DeepSeek-R1 Planner + DeepSeek-V3 elsewhere)

Cumulative token consumption per task across all 5 trials. Counts are taken from the
DeepSeek API `usage` fields, split by model.

| Task | V3 Input | V3 Output | R1 Input | R1 Output | Cost (USD) |
|:-----|---------:|----------:|---------:|----------:|-----------:|
| FB  | 92,370 | 45,132 | 12,505 | 10,464 | $0.1044 |
| AR  | 61,731 | 22,666 | 18,561 | 13,564 | $0.0815 |
| TE  | 50,139 | 15,680 | 16,643 | 14,035 | $0.0707 |
| CP  | 79,553 | 27,834 | 16,191 | 14,999 | $0.0938 |
| ETT | 48,915 | 18,205 | 14,043 | 15,204 | $0.0743 |
| ILI | 53,912 | 17,801 | 20,266 | 19,346 | $0.0877 |
| HW  | 48,194 | 11,659 | 15,431 | 12,793 | $0.0623 |
| EC  | 45,319 | 15,369 | 18,997 | 14,203 | $0.0707 |
| MCS | 64,216 | 18,725 | 16,659 | 19,492 | $0.0898 |
| WBY | 42,371 |  8,595 | 13,085 | 14,655 | $0.0602 |
| ST  | 35,763 |  4,902 | 10,254 | 18,060 | $0.0602 |
| ES  | 54,994 | 18,107 | 14,505 | 21,912 | $0.0907 |
| **Total** | **677,477** | **224,675** | **187,140** | **188,727** | **$0.9463** |

## Token Usage: DeepSeek-V3 (Uniform)

Cumulative token consumption per task across all 5 trials, DeepSeek-V3 in every role.

| Task | V3 Input | V3 Output | Cost (USD) |
|:-----|---------:|----------:|-----------:|
| FB  | 96,651 | 41,436 | $0.0717 |
| AR  | 85,518 | 29,067 | $0.0551 |
| TE  | 93,790 | 34,623 | $0.0634 |
| CP  | 66,871 | 20,064 | $0.0401 |
| ETT | 54,418 | 14,333 | $0.0305 |
| ILI | 98,187 | 26,557 | $0.0557 |
| HW  | 54,564 | 14,308 | $0.0305 |
| EC  | 71,088 | 18,265 | $0.0393 |
| MCS | 83,370 | 22,546 | $0.0473 |
| WBY | 74,278 | 21,287 | $0.0435 |
| ST  | 44,439 |  8,278 | $0.0211 |
| ES  | 77,684 | 21,840 | $0.0450 |
| **Total** | **900,858** | **272,604** | **$0.5431** |

## Cost and Token Summary

Each configuration was run over 60 development runs (12 tasks × 5 trials). Per-run values
are derived from the exact totals above.

| Configuration | Total Tokens | Tokens / Run | Cost / Run | Total Cost |
|:--------------|-------------:|-------------:|-----------:|-----------:|
| Hybrid (R1+V3) | 1,278,019 | 21,300 | $0.0158 | $0.9463 |
| DeepSeek-V3 (uniform) | 1,173,462 | 19,558 | $0.0091 | $0.5431 |
| GPT-4 | *not measured* | *not measured* | $1.6000 | — |

GPT-4 cost is the per-run development figure from Table 4 of Guo et al. (ICML 2024). GPT-4
was never run in this study, so no token counts exist for it.

## Pricing Reference

Rates current at the time of experimentation.

| Model | Input | Output |
|:------|------:|-------:|
| DeepSeek-V3 (`deepseek-chat`) | $0.27 / 1M tokens | $1.10 / 1M tokens |
| DeepSeek-R1 (`deepseek-reasoner`) | $0.55 / 1M tokens | $2.19 / 1M tokens |
