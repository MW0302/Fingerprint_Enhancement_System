# Pipeline Scaffold — Setup

## What's here

```
src/
  utils/config.py       shared paths (dataset location, NFIQ2 exe) — same on every machine
  utils/common.py       shared segmentation / normalisation / orientation-field / NFIQ2 helpers
  pipeline_a/pipeline_a.py   Member A — CLAHE + Gabor
  pipeline_b/pipeline_b.py   Member B — Wavelet + Adaptive Thresholding + Morphology
  pipeline_c/pipeline_c.py   Member C — Coherence Diffusion + 2D Log-Gabor
  pipeline_d/pipeline_d.py   Member D — STFT
dashboard/app.py         Streamlit app wiring all four pipelines + NFIQ2 together
```

Every `pipeline_X.py` has an `enhance(image, params=None)` function — that is
the one function each member needs to finish. See
`Team_Member_Starter_Packets.docx` for what each pipeline needs to target,
which techniques to implement, and the citations to use. The shared steps
(segmentation, normalisation, orientation field, NFIQ2 scoring) are already
implemented in `src/utils/common.py` — reuse them, don't rewrite them.

Every pipeline file also has a `__main__` block at the bottom for a quick
single-image test — run e.g. `python pipeline_a.py` directly to sanity-check
your changes before touching the dashboard.

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

Every pipeline currently runs end-to-end (tested), but each member's core
technique (the parts marked `TODO` in their file) is a placeholder that just
passes the image through unchanged. This is intentional — the goal right now
is that the plumbing (shared utilities, dashboard wiring, NFIQ2 scoring)
already works, so each member's actual job is only to fill in their one
`TODO` section, not to also build the integration around it.
