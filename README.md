# Pipeline Scaffold — Setup

**Updated 30 August 2026 (three times now)** — every pipeline targets the
same three problems (P1 contrast, P2 noise, P6 orientation), each with three
distinct techniques, preceded by a shared normalisation + segmentation
preprocessing step. See `Dataset_Problem_Analysis_and_Revised_Pipelines.md`
(or the `.docx`), Sections 5–6, for the full reasoning — the three-problem
pivot was a lecturer-mandated change so that all four pipelines are directly
comparable and no technique repeats across pipelines; the preprocessing step
was dropped and then brought back (see the "Why shared
segmentation/normalisation is back" note below) for the same
directly-comparable reason. The third update is a further debugging round on
`common.py` and `pipeline_c.py`, driven by full 320-image batch testing
rather than small pilots — see that note below for what changed.

**If you pulled this repo before 30 August and already have a local copy of
`pipeline_a/b/c/d.py` or `common.py`, `git pull`/`Fetch origin` again** —
`common.py`'s `normalize_image()`/`segment()` and every pipeline's `enhance()`
changed in this update. This applies even if you already pulled once on 30
August — `common.py` has changed twice more since then.

## What's here

```
src/
  utils/config.py       shared paths (dataset location, NFIQ2 exe) — same on every machine
  utils/common.py       shared normalisation + segmentation + orientation-field +
                         NFIQ2 helpers — normalize_image() and segment() are called
                         by every pipeline's Step 0 now (see the "Why shared
                         segmentation/normalisation is back" note below)
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

### Why shared segmentation/normalisation is back

Every pipeline used to start with Otsu block-variance segmentation and
Hong et al. block-wise normalisation before its own technique, then this was
dropped (30 August, first revision) because all four pipelines calling the
exact same two functions the exact same way would count as a repeated
technique — and then brought back the same day (second revision), as a
shared **preprocessing** step, not a fourth counted technique.

Two things changed this back:
1. Pipeline C added normalisation + segmentation back for itself first, to
   more closely follow the group's own Literature Review 2 (Shams et al.,
   2023), whose published method runs on an already-conditioned,
   foreground-only image.
2. Giving only Pipeline C a differently-conditioned input than A/B/D would
   have violated the group's own "Fair Experimental Conditions" principle
   (Handover Notes) — so normalisation + segmentation were extended to all
   four pipelines. This doesn't reopen the repeated-technique problem: a
   preprocessing step that conditions the image without independently
   solving P1/P2/P6 itself is treated the same way ridge orientation field
   estimation already is (shared by Pipelines A, B, C without counting as a
   fourth technique for any of them) — see `common.py`'s module docstring.

`segment()` and `normalize_image()` are now called at the top of every
`enhance()`, before each pipeline's own three techniques. Things worth
knowing if you're touching this code:
- `normalize_image()`'s target standard deviation is 40, not Hong et al.'s
  own textbook default of 10 — the lower value was found (testing Pipeline
  A's CLAHE step) to crush contrast so much that CLAHE couldn't recover it.
  If your own technique is sensitive to input contrast, it's worth testing
  explicitly rather than assuming this default is neutral for you too.
- **Changed again (third update):** `normalize_image()` used to force
  *every* image's variance toward that std=40 target, even images whose raw
  contrast already exceeded it — full-batch testing (on DB1, the highest raw
  NFIQ2 subset) showed this was actually reducing contrast on already-good
  images, and an intermediate buggy version of the fix was also clipping a
  meaningful fraction of pixels to pure black on some high-mean images.
  `normalize_image()` now passes an image through completely unchanged (no
  mean shift, no variance rescale) if its raw variance already meets or
  exceeds the target — only images below the target get boosted, same as
  before. If your own step was written around the "always rescales" old
  behaviour, re-check it: an image that used to arrive at your step
  compressed to std≈40 may now arrive at its original, higher contrast.
- `segment()`'s foreground mask (`fg_mask_blocks`) is available to your
  Step 2/3 implementation if you want to skip or de-emphasise background
  blocks, but using it is optional — it's there if it helps your technique,
  not a requirement. Its hole-filling logic (third update) now uses a
  convex-hull fix instead of a small morphological closing — the closing
  couldn't fill larger holes (e.g. a whole misclassified ridge core on some
  DB3 images); the convex-hull version can. This only changes `segment()`'s
  internals, not how you call it.

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

Every pipeline's shared Step 0 (normalisation + segmentation) is
implemented. Pipeline C is fully implemented and validated against the full
320-image dataset (not just a pilot subset), including a quality-adaptive
scheme that scales its technique strength per image based on estimated
ridge-orientation coherence — see the Status note in
`Team_Member_Starter_Packets.docx` (Member C section) for the mechanism and
the reasoning, and for why DB1 still shows a net-negative result even so
(deliberate trade-off, not an unfixed bug). Final validated full-batch NFIQ2
deltas: DB1_B −8.25 (80% regressed), DB2_B +7.99 (74% improved), DB3_B
+13.78 (87% improved), DB4_B +12.21 (90% improved). Pipeline A's Step 1
(CLAHE) is implemented as a starting point.
Everything else — Pipeline A's Steps 2-3, and all of Pipelines B and D — runs
end-to-end (the plumbing works) but each `TODO`-marked technique is still a
placeholder that passes the image through unchanged. That's intentional —
each member's job is to research, implement, and be able to explain their
own pipeline's TODO steps. Citations aren't pre-supplied for any technique
except Pipeline C's Log-Gabor step (Shams et al., 2023, already established
as LR2 for the group) — find and verify your own citations for whatever you
implement rather than reusing anything you see referenced elsewhere in this
scaffold.
