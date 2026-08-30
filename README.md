# Pipeline Scaffold — Setup

**Updated 30 August 2026** — the shared segmentation/normalisation module has
been dropped and every pipeline now targets the same three problems (P1
contrast, P2 noise, P6 orientation), each with three distinct techniques.
See `Dataset_Problem_Analysis_and_Revised_Pipelines.md` (or the `.docx`),
Sections 5–6, for the full reasoning — this was a lecturer-mandated pivot so
that all four pipelines are directly comparable and no technique repeats
across pipelines.

## What's here

```
src/
  utils/config.py       shared paths (dataset location, NFIQ2 exe) — same on every machine
  utils/common.py       shared orientation-field + NFIQ2 helpers
                         (segment() and normalize_image() still exist here
                         but are no longer called by any pipeline — see the
                         "Why no shared segmentation/normalisation" note below)
  pipeline_a/pipeline_a.py   Member A — CLAHE + Median/Bilateral + Gabor
  pipeline_b/pipeline_b.py   Member B — Wavelet Contrast + Wavelet Denoising + Morphology
  pipeline_c/pipeline_c.py   Member C — Homomorphic + Coherence Diffusion + Log-Gabor (done)
  pipeline_d/pipeline_d.py   Member D — FFT Emphasis + Wiener/Notch + STFT
dashboard/app.py         Streamlit app wiring all four pipelines + NFIQ2 together
```

Every `pipeline_X.py` has an `enhance(image, params=None)` function — that is
the one function each member needs to finish. See
`Team_Member_Starter_Packets.docx` for what each pipeline needs to target and
which techniques to implement. Each pipeline file's own docstring also lists
exactly which steps are already implemented as a starting point versus left
as a `TODO` for you to research and implement. Pipeline C (mine) is fully
implemented already — read it if you want to see what a finished
`enhance()` looks like, but your own pipeline's core techniques still need
to be your own work, not a copy of mine.

Every pipeline file also has a `__main__` block at the bottom for a quick
single-image test — run e.g. `python pipeline_a.py` directly to sanity-check
your changes before touching the dashboard.

### Why no shared segmentation/normalisation anymore

Every pipeline used to start with Otsu block-variance segmentation and
Hong et al. block-wise normalisation before its own technique. Those
targeted P5 (variable background) and P7 (within-subset heterogeneity),
which are no longer problems any pipeline is trying to solve — and since
all four pipelines would otherwise be calling the exact same two functions
in the exact same way, keeping them would count as a repeated technique
across pipelines, which the lecturer explicitly does not want. `segment()`
and `normalize_image()` are still in `common.py` in case you want them for
your own experimentation, but they're not part of the required pipeline
anymore — don't call them in your `enhance()` unless you have a specific
reason tied to your own technique.

---

## 一次性设置(团队所有人都要看)

### 1. 安装Python库

VS Code里打开终端(用你之前配好的Anaconda解释器),跑:

```
pip install streamlit PyWavelets scikit-image
```

(`opencv-python`、`numpy`、`pandas`应该在之前跑`analyze_dataset.py`的时候就装过了。)

### 2. 装GitHub Desktop(不用学命令行)

下载:https://desktop.github.com — 装好后用GitHub账号登录(没有账号的话先在github.com免费注册一个)。

### 3. 拿到代码

**Leader(你)第一次要做:**
1. 上github.com,右上角"+"→"New repository",取个名字(比如`fingerprint-enhancement`),选**Private**,不要勾选任何初始化选项,直接Create。
2. Settings → Collaborators → Add people,把3个队友的GitHub用户名/邮箱加进去,他们各自会收到邀请邮件,接受就行。
3. 在GitHub Desktop里:File → Clone repository,选你刚建的这个空repo,clone到电脑上你喜欢的位置。
4. 把我发给你的这整个文件夹(`src/`、`dashboard/`、`README.md`、`.gitignore`)复制进刚clone出来的那个文件夹里。
5. 回到GitHub Desktop,左边会看到一堆新增文件,下方写个commit message(比如"initial pipeline scaffold"),点`Commit to main`,再点右上角`Push origin`。

**队友(4人都要做,包括你自己):**
1. GitHub Desktop → File → Clone repository → 输入/选择leader建的那个repo → clone到自己电脑。
2. 在自己电脑上,把之前下载好的4个FVC2002 zip解压到clone下来的文件夹里的 `data/raw/` 下面,结构要是:
   ```
   <你clone的文件夹>/data/raw/DB1_B/*.tif
   <你clone的文件夹>/data/raw/DB2_B/*.tif
   <你clone的文件夹>/data/raw/DB3_B/*.tif
   <你clone的文件夹>/data/raw/DB4_B/*.tif
   ```
   注意:`data/`这个文件夹已经在`.gitignore`里,不会被传上GitHub——每个人都要自己在本地放一份数据集,数据集本身不进git(320张图太大,也没必要共享,大家用的是同一份公开数据集)。
3. 装NFIQ2:跟leader之前一样的步骤,官方MSI装到`C:\Program Files\NFIQ 2\bin\nfiq2.exe`(默认路径,装完不用改代码)。如果你只是在写自己的pipeline、还没到要看NFIQ2分数的阶段,这一步可以先跳过,晚点再装。

### 4. 日常怎么改代码 / 交回来

- 每次开始改代码前,先在GitHub Desktop点一下`Fetch origin`(如果有新提交会变成`Pull origin`,点一下拉下来),确保自己是最新版本。
- 打开自己负责的`pipeline_X.py`,把TODO那段实现出来,存档。
- 回到GitHub Desktop,能看到改动了哪些文件、具体改了什么(左右对比),下方写个commit message(比如"implement CLAHE + Gabor step"),`Commit to main`,然后`Push origin`。
- 因为每人只改自己的`pipeline_X.py`,基本不会跟别人冲突。如果以后要一起改`dashboard/app.py`,建议提前群里说一声"我要改dashboard了"避免同时改。

Leader这边定期点`Fetch origin`/`Pull origin`,就能拿到所有人推上来的最新代码。

---

## Running the dashboard

From the `dashboard` folder:

```
streamlit run app.py
```

This opens in your browser automatically.

## Current state

Pipeline C is fully implemented and tested across all four subsets.
Pipelines A, B, and D run end-to-end (the plumbing works), but each one's
`TODO`-marked techniques are still placeholders that pass the image through
unchanged. That's intentional — each member's job is to research, implement,
and be able to explain their own pipeline's TODO steps. Citations aren't
pre-supplied for any technique except Pipeline C's Log-Gabor step (Shams et
al., 2023, already established as LR2 for the group) — find and verify your
own citations for whatever you implement rather than reusing anything you
see referenced elsewhere in this scaffold.
