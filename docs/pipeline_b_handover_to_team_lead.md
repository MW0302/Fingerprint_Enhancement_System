# Pipeline B 交接说明

## 当前状态

- 负责成员：Member B
- 工作分支：`feature/pipeline-b`
- 合并目标：`main`
- 状态：三个 technique、pilot tests、320 张全量 cumulative ablation、最终 CSV、dashboard results loader 和英文报告初稿均已完成
- 参数状态：已锁定；全量 320 张结果仅用于验证，没有用来重新调参

## Pipeline B 已实现内容

Pipeline B 使用 wavelet / multiresolution 技术路线，并严格保留三个 counted techniques：

1. P1 Contrast：wavelet decomposition + detail-coefficient contrast enhancement
2. P2 Noise：noise-adaptive BayesShrink wavelet shrinkage denoising
3. P6 Orientation：ridge orientation field + orientation-steered greyscale morphological processing

共用的 `normalize_image()`、`segment()` 和 `orientation_field()` 均直接调用 `src/utils/common.py`，没有另外复制一套。最终 `enhance()` 返回原尺寸、8-bit greyscale image；没有把 binary image 交给 NFIQ2。

三个 counted techniques 已拆成独立 internal functions，支持队长要求的 cumulative ablation：

```text
Raw
 -> Stage 1: P1
 -> Stage 2: P1 + P2
 -> Stage 3: P1 + P2 + P6
```

## 锁定参数

### P1

- wavelet：`db4`
- decomposition level：3
- coarsest/final detail gain：1.60 / 1.00
- coefficient reliability floor：25th percentile
- approximation coefficients：unchanged

### P2

- wavelet：`db4`, level 3
- threshold：BayesShrink soft threshold
- processed levels：finest 1 level
- noise-adaptive factor：`clip((sigma / 5)^4, 0.10, 1.00)`

### P6

- line kernel length：7 pixels
- orientation bins：12
- strength：0.50
- coherence floor/power：0.20 / 1.00
- maximum pre-strength darkening：16 grey levels
- adjacent directional responses are interpolated with doubled-angle orientation handling

## Pilot testing

同一组固定 16 张样本被用于三个连续 pilot，每个 DB 4 张：

- P1：21 candidates
- P2：28 candidates，P1 参数保持锁定
- P6：30 candidates，P1/P2 参数保持锁定

候选选择同时考虑 mean gain、regression count、worst regression 和 visual quality；没有机械选择 mean gain 最高但风险更大的 candidate。

## 最终 320 张结果

数据完整性：

- DB1_B：80
- DB2_B：80
- DB3_B：80
- DB4_B：80
- 总计：320
- Stage 3 与正式 `enhance()` pixel-identical：320/320
- Stage 1、Stage 2、Stage 3 均取得 320 个有效 NFIQ2 scores

整体 marginal contribution：

| Increment | Mean delta | Improved | Regressed | Unchanged |
|---|---:|---:|---:|---:|
| P1: Stage 1 - Raw | +4.931 | 218 | 79 | 22 |
| P2: Stage 2 - Stage 1 | +0.706 | 124 | 88 | 108 |
| P6: Stage 3 - Stage 2 | +1.034 | 175 | 121 | 24 |
| Final: Stage 3 - Raw | +6.674 | 237 | 73 | 9 |

P1 和 Final 使用 319 张 paired images，原因见下方 NFIQ2 exception。P2 和 P6 使用全部 320 张 stage-to-stage comparisons。

分数据库 final mean delta：

| Database | Final mean delta |
|---|---:|
| DB1_B | +1.813 |
| DB2_B | +7.388 |
| DB3_B | +11.418 |
| DB4_B | +6.138 |

Raw quality 与 final improvement 呈负相关：Pearson `-0.408`。最低 raw-quality quartile 平均提升 `+10.325`，最高 quartile 只提升 `+0.636`。因此这条 pipeline 对 degraded fingerprints 的帮助明显高于 already-high-quality fingerprints。

## NFIQ2 exception

`DB3_B/110_5.tif` 的 raw image 被 NFIQ2 拒绝：

```text
FRFXLL_ERR_FB_TOO_SMALL_AREA: Fingerprint area is too small.
```

这不是 Pipeline B output failure；Pipeline C 的最终 CSV 对同一 raw image 也没有 raw score。Pipeline B 的 Stage 1/2/3 仍成功得到 `14 / 18 / 15`。CSV 保留空白 raw score 和完整官方 error，没有填假值。

## Visual audit

已检查最差 8 张 final regressions 和固定随机抽取的 8 张样本。没有观察到：

- cross-ridge merging
- wrong-direction strokes
- block seams
- background spill
- binary output
- greyscale clipping

最大 regression 是 `DB1_B/101_6.tif`：`86 -> 66`。Regression 主要集中于 raw NFIQ2 已经很高的干净图像，已在报告 limitations/discussion 中诚实记录，没有根据全量结果回头调参。

## 重要文件

- Pipeline implementation：`src/pipeline_b/pipeline_b.py`
- P1 pilot：`scripts/pipeline_b_p1_pilot.py`
- P2 pilot：`scripts/pipeline_b_p2_pilot.py`
- P6 pilot：`scripts/pipeline_b_p3_pilot.py`
- Full ablation：`scripts/pipeline_b_ablation.py`
- Visual audit：`scripts/pipeline_b_visual_audit.py`
- Report statistics：`scripts/pipeline_b_report_stats.py`
- Per-image final CSV：`results/pipeline_b_ablation.csv`
- Summary CSV：`results/pipeline_b_ablation_summary.csv`
- Regression CSV：`results/pipeline_b_worst_regressions.csv`
- English report draft：`docs/pipeline_b_report_draft.md`
- Method notes：`docs/pipeline_b_p1_method_notes.md`、`docs/pipeline_b_p2_method_notes.md`、`docs/pipeline_b_p3_method_notes.md`

## 验证结果

- Pipeline B unit tests：16/16 passed
- Dataset manifest：320 rows、80 rows per DB
- Raw scores available：319/320
- Stage 1/2/3 scores available：320/320
- Stage 3 equals `enhance()`：320/320
- Dashboard：已能从 final CSV 加载 Pipeline B 320 条 results
- Git diff check：passed

复现命令：

```powershell
conda activate fingerprint-b
python -m unittest discover -s tests -v
python scripts/pipeline_b_ablation.py --smoke
python scripts/pipeline_b_ablation.py --full
python scripts/pipeline_b_report_stats.py
```

`--full` 支持 checkpoint/resume，并拒绝在 manifest、code hash、NFIQ2 executable 或 frozen parameters 不一致时错误续跑。

## 请队长 review 的项目

1. 确认三个 technique 与最终 mapping 一致，没有与其他 pipeline 重复。
2. 确认 cumulative CSV columns 能直接接入最终 hybrid-selection script。
3. 确认 `DB3_B/110_5.tif` 保留 missing Raw score 的处理与 Pipeline C 一致。
4. 将英文报告初稿中的 section/figure numbering 调整到 group report 的最终结构。
5. 等 Pipeline A/C/D 的最终 cumulative results 都统一后，再以 data-driven 方法选择固定的 hybrid P1/P2/P6；不要根据单张 image 动态切换 technique。

## 本分支 commits

```text
81695e1 Implement Pipeline B wavelet contrast stage
75df8b8 Implement Pipeline B wavelet shrinkage stage
b12ca1e Implement Pipeline B oriented morphology stage
f956740 Add Pipeline B full ablation results
959561d Add Pipeline B report draft
```
