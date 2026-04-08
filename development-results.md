# DS-Agent | Development Stage Results

This document consolidates all benchmark results from the **Development Stage** of DS-Agent across multiple model configurations. Each task is evaluated over **5 trials**; `F` denotes a failed trial.

---

## Table of Contents

1. [Model Comparison Summary](#1-model-comparison-summary)
2. [Run Logs | DeepSeek Models](#2-run-logs--deepseek-models)
3. [Run Logs | GPT Models](#3-run-logs--gpt-models)
4. [Token Usage | DeepSeek Hybrid (V3 + R1)](#4-token-usage--deepseek-hybrid-v3--r1)
5. [Token Usage | DeepSeek Single (V3 Only)](#5-token-usage--deepseek-single-v3-only)

---

## 1. Model Comparison Summary

Final best-trial scores for each task across all model configurations. The **Winner** column identifies the top-performing model per task.

| Task | Metric | DS GPT-3.5 | DS GPT-4 | DS DeepSeek-V3 | DS DeepSeek-R1-V3 | Winner |
|------|--------|:----------:|:--------:|:--------------:|:-----------------:|--------|
| FB  | MCRMSE ↓ | 0.6155 | 0.5277 | 0.5596 | **0.5186** | DeepSeek-R1-V3 |
| AR  | RMSE ↓   | 1.6926 | 1.5016 | 1.5345 | **1.3135** | DeepSeek-R1-V3 |
| TE  | Accuracy ↑ | 0.7830 | 0.7976 | 0.8490 | **0.8529** | DeepSeek-R1-V3 |
| CP  | Accuracy ↑ | 0.9760 | 0.9846 | 0.9812 | **0.9847** | DeepSeek-R1-V3 |
| ETT | MSE ↓    | 0.5073 | **0.3222** | 0.3782 | 0.5840 | GPT-4 |
| ILI | MSE ↓    | 5.2060 | 3.5371 | 3.5825 | **3.3182** | DeepSeek-R1-V3 |
| HW  | Accuracy ↑ | 0.3352 | 0.2423 | 0.5494 | **0.5553** | DeepSeek-R1-V3 |
| EC  | AUROC ↑  | 0.3536 | **0.6958** | 0.3308 | 0.3422 | GPT-4 |
| MCC | RMSLE ↓  | **0.1968** | 0.2858 | 0.2947 | 0.2892 | GPT-3.5 |
| WBY | MAE ↓    | **180** | 182 | 313.63 | 276.33 | GPT-3.5 |
| ST  | AUROC ↑  | 0.8027 | 0.8062 | 0.8977 | **0.8978** | DeepSeek-R1-V3 |
| ES  | AUROC ↑  | 0.7031 | **0.9761** | 0.9309 | 0.9188 | GPT-4 |

> ↓ = lower is better &nbsp;&nbsp;|&nbsp;&nbsp; ↑ = higher is better

---

## 2. Run Logs | DeepSeek Models

All 5 trial results for **DS-Agent** using DeepSeek-V3 and DeepSeek-R1-V3. Values are listed in trial order (Trial 1 → Trial 5).

| Task | Metric | DS-Agent (DeepSeek-V3) | DS-Agent (DeepSeek-R1-V3) |
|------|--------|------------------------|---------------------------|
| FB  | MCRMSE ↓ | F, F, 0.5596, 0.5718, 0.5604 | 0.5767, 0.5545, 0.5533, 0.5324, **0.5186** |
| AR  | RMSE ↓   | 1.6550, 1.6702, 1.5346, F, 1.5345 | 1.6550, 1.4397, 1.4277, 1.3719, **1.3135** |
| TE  | Accuracy ↑ | 0.6661, 0.8178, 0.8427, F, 0.8490 | 0.6661, 0.8213, 0.8272, **0.8529**, **0.8529** |
| CP  | Accuracy ↑ | 0.9436, 0.9761, 0.9761, 0.9778, 0.9812 | 0.9692, 0.9795, 0.9812, **0.9846**, **0.9846** |
| ETT | MSE ↓    | 0.5369, 0.4455, 0.4710, **0.3782**, 0.3855 | 1.1102, 1.0385, 0.9311, 0.6824, 0.5840 |
| ILI | MSE ↓    | 4.5930, 5.1587, 3.8230, 3.7345, 3.5825 | 4.1982, 3.9816, 3.8244, 3.5232, **3.3182** |
| HW  | Accuracy ↑ | 0.0365, 0.0953, 0.2306, 0.5400, 0.5494 | 0.1059, 0.2859, 0.5294, 0.5400, **0.5553** |
| EC  | AUROC ↑  | 0.3308, 0.2776, 0.2624, 0.2624, 0.2814 | 0.3308, 0.3346, 0.3346, 0.3346, **0.3422** |
| MCC | RMSLE ↓  | 0.3024, 0.2957, 0.2944, 0.2957, 0.2947 | 0.3021, 0.2938, 0.2915, 0.2901, **0.2892** |
| WBY | MAE ↓    | F, 331.29, 331.56, 327.53, 313.63 | 310.59, 301.37, 288.83, 288.83, **276.33** |
| ST  | AUROC ↑  | **0.8977**, 0.8832, 0.8832, 0.8773, 0.8813 | 0.8794, 0.8859, 0.8919, 0.8925, **0.8978** |
| ES  | AUROC ↑  | **0.9188**, 0.7643, 0.7240, 0.7603, **0.9309** | 0.7365, 0.7527, 0.7946, 0.8365, 0.9188 |

---

## 3. Run Logs | GPT Models

All 5 trial results for **ResearchAgent** and **DS-Agent** using GPT-3.5 and GPT-4. Values are listed in trial order (Trial 1 → Trial 5).

| Task | Metric | ResearchAgent (GPT-3.5) | ResearchAgent (GPT-4) | DS-Agent (GPT-3.5) | DS-Agent (GPT-4) |
|------|--------|:----------------------:|:--------------------:|:-----------------:|:----------------:|
| FB  | MCRMSE ↓ | F, F, F, F, F | F, F, F, F, 0.6326 | F, F, F, F, 0.6155 | 0.6352, 0.6036, 0.5971, 0.5384, **0.5277** |
| AR  | RMSE ↓   | F, F, F, F, F | F, F, F, 4.4697, 2.1858 | F, F, F, 4.7162, 1.6926 | 2.7393, 2.4784, 1.8065, 1.6982, **1.5016** |
| TE  | Accuracy ↑ | F, F, F, F, F | F, F, F, 0.3451, 0.2504 | F, 0.3154, 0.6544, 0.7650, 0.7830 | 0.2504, 0.3239, 0.6452, 0.6640, **0.7976** |
| CP  | Accuracy ↑ | F, F, F, F, F | F, F, 0.6051, 0.7401, 0.9145 | F, 0.9282, 0.9435, 0.9555, 0.9760 | 0.9162, 0.9487, 0.9743, 0.9811, **0.9846** |
| ETT | MSE ↓    | F, F, F, F, 1.0347 | F, F, F, F, F | F, F, 2.1335, 0.8316, 0.5073 | 2.7290, 0.8566, 0.7363, 0.4071, **0.3222** |
| ILI | MSE ↓    | F, F, F, F, F | F, F, F, F, 4.4214 | F, 5.4091, 5.2496, 5.2124, 5.2060 | 5.5217, 4.6283, 4.5704, 4.2758, **3.5371** |
| HW  | Accuracy ↑ | F, F, F, 0.0333, 0.0411 | F, F, F, 0.0470, 0.0470 | 0.0411, 0.0423, 0.0670, 0.2929, **0.3352** | 0.0482, 0.0623, 0.0952, 0.1952, 0.2423 |
| EC  | AUROC ↑  | F, F, F, F, 0.3460 | F, F, 0.3688, 0.3954, 0.4144 | F, F, 0.3003, 0.3498, 0.3536 | 0.2585, 0.2661, 0.3688, 0.3802, **0.6958** |
| MCC | RMSLE ↓  | F, F, F, F, F | F, 0.3168, 0.3168, 0.3168, 0.1774 | 0.3106, 0.2981, 0.2957, 0.2498, **0.1968** | 0.3020, 0.2974, 0.2970, 0.2916, 0.2858 |
| WBY | MAE ↓    | F, F, F, F, F | 6029, 371, 371, 371, **180** | 371, 307, 300, 299, **180** | 325, 325, 325, 182, 182 |
| ST  | AUROC ↑  | F, F, F, F, 0.4956 | 0.7688, 0.7688, 0.7688, 0.7906, 0.7970 | F, 0.7970, 0.7970, 0.8016, 0.8027 | 0.7780, 0.7993, 0.8027, 0.8027, **0.8062** |
| ES  | AUROC ↑  | F, F, F, F, F | F, F, 0.5394, 0.8569, 0.9751 | 0.4343, 0.4981, 0.4981, 0.6338, 0.7031 | 0.6258, 0.6258, 0.9576, **0.9761**, **0.9761** |

---

## 4. Token Usage | DeepSeek Hybrid (V3 + R1)

Token consumption and estimated cost per task when using the **Hybrid** configuration (DeepSeek-V3 for code generation + DeepSeek-R1 for reasoning).

| Task | V3 Input Tokens | V3 Output Tokens | R1 Input Tokens | R1 Output Tokens | Cost (USD) |
|------|----------------:|-----------------:|----------------:|-----------------:|:----------:|
| FB  | 92,370 | 45,132 | 12,505 | 10,464 | $0.1044 |
| AR  | 61,731 | 22,666 | 18,561 | 13,564 | $0.0815 |
| TE  | 50,139 | 15,680 | 16,643 | 14,035 | $0.0707 |
| CP  | 79,553 | 27,834 | 16,191 | 14,999 | $0.0938 |
| ETT | 48,915 | 18,205 | 14,043 | 15,204 | $0.0743 |
| ILI | 53,912 | 17,801 | 20,266 | 19,346 | $0.0877 |
| HW  | 48,194 | 11,659 | 15,431 | 12,793 | $0.0623 |
| EC  | 45,319 | 15,369 | 18,997 | 14,203 | $0.0707 |
| MCC | 64,216 | 18,725 | 16,659 | 19,492 | $0.0898 |
| WBY | 42,371 | 8,595  | 13,085 | 14,655 | $0.0602 |
| ST  | 35,763 | 4,902  | 10,254 | 18,060 | $0.0602 |
| ES  | 54,994 | 18,107 | 14,505 | 21,912 | $0.0907 |
| **Total** | | | | | **$0.9463** |

### Pricing Reference

| Model | Input | Output |
|-------|------:|-------:|
| DeepSeek-V3 | $0.27 / 1M tokens | $1.10 / 1M tokens |
| DeepSeek-R1 | $0.55 / 1M tokens | $2.19 / 1M tokens |

---

## 5. Token Usage | DeepSeek Single (V3 Only)

Token consumption and estimated cost per task when using **DeepSeek-V3 alone** (no reasoning model).

| Task | V3 Input Tokens | V3 Output Tokens | Cost (USD) |
|------|----------------:|-----------------:|:----------:|
| FB  | 96,651 | 41,436 | $0.0717 |
| AR  | 85,518 | 29,067 | $0.0551 |
| TE  | 93,790 | 34,623 | $0.0634 |
| CP  | 66,871 | 20,064 | $0.0401 |
| ETT | 54,418 | 14,333 | $0.0305 |
| ILI | 98,187 | 26,557 | $0.0557 |
| HW  | 54,564 | 14,308 | $0.0305 |
| EC  | 71,088 | 18,265 | $0.0393 |
| MCC | 83,370 | 22,546 | $0.0473 |
| WBY | 74,278 | 21,287 | $0.0435 |
| ST  | 44,439 | 8,278  | $0.0211 |
| ES  | 77,684 | 21,840 | $0.0450 |
| **Total** | | | **$0.5431** |

### Pricing Reference

| Model | Input | Output |
|-------|------:|-------:|
| DeepSeek-V3 | $0.27 / 1M tokens | $1.10 / 1M tokens |

---

> **Note:** The Hybrid configuration (V3 + R1) costs approximately **$0.9463** vs **$0.5431** for V3-only. A ~74% cost increase in exchange for improved reasoning quality on complex tasks.
