"""
Loads already-computed NFIQ2 results per pipeline from disk, so the
Overview/Comparison tab doesn't have to re-run a full 320-image batch
through Streamlit every time it's opened -- that would take a long time and
duplicate work someone already did on their own machine. Tolerant of
missing files/columns per pipeline/DB: reports what's available instead of
crashing, and never fabricates a number that wasn't actually computed.

Result locations, checked directly against what's actually on disk (not
assumed -- see each loader's docstring for how it was confirmed):

    Pipeline A: data/pipeline_a_full_320_cumulative_nfiq2_config_*/
                per_image_cumulative_scores.csv -- has raw/stage1/stage2/
                stage3 NFIQ2 per image already.
    Pipeline B: no final 320-image persisted result yet. P1 and P2 are
                implemented and have local pilot evidence, while P6 remains
                TODO; tuning outputs are intentionally not treated as final.
    Pipeline C: data/processed/pipeline_c/<DB>/batch_results.csv (raw +
                final-enhanced NFIQ2 per DB) is the validated production
                source. results/pipeline_c_ablation.csv additionally has
                per-stage scores when present, used opportunistically for
                richer detail but not required.
    Pipeline D: results/pipeline_d_ablation/pipeline_d_ablation.csv, IF that
                run has been done and its output kept locally. As of this
                writing nothing is committed for Pipeline D (only the
                scripts/pipeline_d_ablation.py tool itself is in git --
                results/ and data/processed/ are gitignored, the same reason
                none of Pipeline C's local sweep data is shared either) --
                so Pipeline D currently has NOTHING for this loader to find,
                contrary to it being described as already "fully validated
                with results available". It will show the same "not yet
                available, live-run only" state as Pipeline B until someone
                runs pipeline_d_ablation.py and the output lands on disk.

Every load_pipeline_*() function returns a DataFrame with at least the
columns `file`, `db`, `raw_nfiq2`, `enhanced_nfiq2` (plus optional
`stage1_nfiq2`/`stage2_nfiq2` where the source has them), or None if nothing
was found for that pipeline at all.
"""

import glob
import os

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

PIPELINE_LABELS = {
    "pipeline_a": "Pipeline A",
    "pipeline_b": "Pipeline B",
    "pipeline_c": "Pipeline C",
    "pipeline_d": "Pipeline D",
}


def _newest_matching_dir(pattern):
    """Returns the most-recently-modified directory matching a glob
    pattern under DATA_DIR, or None. Used for Pipeline A's date-stamped
    result folder name so a later re-run with a new date doesn't silently
    stop being picked up."""
    candidates = [p for p in glob.glob(os.path.join(DATA_DIR, pattern)) if os.path.isdir(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_pipeline_a():
    """Pipeline A: data/pipeline_a_full_320_cumulative_nfiq2_config_*/
    per_image_cumulative_scores.csv -- confirmed present with columns
    database, filename, raw_nfiq2, stage1_nfiq2, stage2_nfiq2, stage3_nfiq2
    (plus per-stage delta columns this loader doesn't need, since the
    Overview tab computes deltas itself from raw/enhanced directly)."""
    folder = _newest_matching_dir("pipeline_a_full_320_cumulative_nfiq2_config_*")
    if folder is None:
        return None
    path = os.path.join(folder, "per_image_cumulative_scores.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    required = {"database", "filename", "raw_nfiq2", "stage3_nfiq2"}
    if not required.issubset(df.columns):
        return None
    out = pd.DataFrame({
        "file": df["filename"],
        "db": df["database"],
        "raw_nfiq2": df["raw_nfiq2"],
        "enhanced_nfiq2": df["stage3_nfiq2"],
    })
    if "stage1_nfiq2" in df.columns:
        out["stage1_nfiq2"] = df["stage1_nfiq2"]
    if "stage2_nfiq2" in df.columns:
        out["stage2_nfiq2"] = df["stage2_nfiq2"]
    return out


def load_pipeline_b():
    """Pipeline B has no final persisted 320-image result yet.

    P1 and P2 are implemented, but their ignored 16-image pilot outputs are
    tuning evidence rather than a final Pipeline B result. P6 is still TODO,
    so returning None remains the expected state.
    """
    return None


def load_pipeline_c():
    """Pipeline C: prefers results/pipeline_c_ablation.csv (has per-stage
    detail: stage1_contrast_nfiq2, stage2_noise_nfiq2,
    stage3_orientation_nfiq2 -- confirmed these are the real column names by
    reading scripts/pipeline_c_ablation.py), falling back to
    data/processed/pipeline_c/<DB>/batch_results.csv (raw_nfiq2,
    enhanced_nfiq2 only, no per-stage detail) if the ablation file isn't
    present. Both are legitimate, already-validated sources for the same
    underlying enhance() output -- see this project's own dual-consistency
    checks confirming their Δtotal always matches."""
    ablation_path = os.path.join(RESULTS_DIR, "pipeline_c_ablation.csv")
    if os.path.isfile(ablation_path):
        try:
            df = pd.read_csv(ablation_path)
        except Exception:
            df = None
        if df is not None and {"file", "db", "raw_nfiq2", "stage3_orientation_nfiq2"}.issubset(df.columns):
            out = pd.DataFrame({
                "file": df["file"],
                "db": df["db"],
                "raw_nfiq2": df["raw_nfiq2"],
                "enhanced_nfiq2": df["stage3_orientation_nfiq2"],
            })
            if "stage1_contrast_nfiq2" in df.columns:
                out["stage1_nfiq2"] = df["stage1_contrast_nfiq2"]
            if "stage2_noise_nfiq2" in df.columns:
                out["stage2_nfiq2"] = df["stage2_noise_nfiq2"]
            return out

    rows = []
    for db in ("DB1_B", "DB2_B", "DB3_B", "DB4_B"):
        batch_path = os.path.join(PROCESSED_DIR, "pipeline_c", db, "batch_results.csv")
        if not os.path.isfile(batch_path):
            continue
        try:
            db_df = pd.read_csv(batch_path)
        except Exception:
            continue
        if {"file", "db", "raw_nfiq2", "enhanced_nfiq2"}.issubset(db_df.columns):
            rows.append(db_df[["file", "db", "raw_nfiq2", "enhanced_nfiq2"]])
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def load_pipeline_d():
    """Pipeline D: results/pipeline_d_ablation/pipeline_d_ablation.csv, the
    default output path scripts/pipeline_d_ablation.py writes to (confirmed
    by reading that script) -- IF it has actually been run and its output
    kept locally. Not present as of this writing (see module docstring)."""
    path = os.path.join(RESULTS_DIR, "pipeline_d_ablation", "pipeline_d_ablation.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # Column names aren't confirmed (the file has never existed locally to
    # inspect) -- try the most likely candidates defensively rather than
    # assuming, so this starts working the moment the file appears without
    # needing a code change, but never crashes if the guess is wrong.
    file_col = next((c for c in ("file", "filename") if c in df.columns), None)
    db_col = next((c for c in ("db", "database") if c in df.columns), None)
    raw_col = next((c for c in ("raw_nfiq2",) if c in df.columns), None)
    enhanced_col = next(
        (c for c in ("enhanced_nfiq2", "stage3_nfiq2", "final_nfiq2") if c in df.columns), None
    )
    if not all((file_col, db_col, raw_col, enhanced_col)):
        return None
    return pd.DataFrame({
        "file": df[file_col],
        "db": df[db_col],
        "raw_nfiq2": df[raw_col],
        "enhanced_nfiq2": df[enhanced_col],
    })


_LOADERS = {
    "pipeline_a": load_pipeline_a,
    "pipeline_b": load_pipeline_b,
    "pipeline_c": load_pipeline_c,
    "pipeline_d": load_pipeline_d,
}


def load_all():
    """Returns {pipeline_key: DataFrame|None} for every pipeline. None means
    genuinely nothing was found on disk -- callers should show that
    explicitly (e.g. "not yet available") rather than silently omitting it,
    so it stays clear which parts of the comparison are backed by real,
    already-validated NFIQ2 runs versus still missing."""
    return {key: loader() for key, loader in _LOADERS.items()}


def availability_summary(loaded=None):
    """Returns a small per-pipeline status DataFrame (n images, source) for
    display -- makes it obvious at a glance what's real data vs. missing,
    instead of only being implicit in which columns are blank."""
    loaded = loaded if loaded is not None else load_all()
    rows = []
    for key, label in PIPELINE_LABELS.items():
        df = loaded.get(key)
        if df is None or len(df) == 0:
            rows.append(dict(pipeline=label, status="not yet available", n_images=0))
        else:
            rows.append(dict(pipeline=label, status="loaded from disk", n_images=len(df)))
    return pd.DataFrame(rows)


def build_master_table(loaded=None):
    """Image | DB | Raw NFIQ2 | Pipeline A | Pipeline B | Pipeline C |
    Pipeline D | Hybrid. Hybrid is intentionally left NaN -- it depends on
    Pipeline B finishing first (see task context), never fabricated here.
    Raw NFIQ2 is taken from whichever pipeline's data has it for that image
    (raw scoring doesn't depend on which pipeline processed the image, so
    any available source is equally valid); if two sources disagree on the
    same image's raw score, that's flagged rather than silently picking
    one, since it would indicate a real inconsistency worth investigating."""
    loaded = loaded if loaded is not None else load_all()

    all_keys = pd.DataFrame(columns=["file", "db"])
    raw_frames = []
    enhanced = {}
    for key in ("pipeline_a", "pipeline_b", "pipeline_c", "pipeline_d"):
        df = loaded.get(key)
        if df is None or len(df) == 0:
            continue
        all_keys = pd.concat([all_keys, df[["file", "db"]]], ignore_index=True)
        raw_frames.append(df[["file", "db", "raw_nfiq2"]])
        enhanced[key] = df.set_index(["file", "db"])["enhanced_nfiq2"]

    if all_keys.empty:
        return pd.DataFrame(columns=[
            "Image", "DB", "Raw NFIQ2", "Pipeline A", "Pipeline B",
            "Pipeline C", "Pipeline D", "Hybrid",
        ]), []

    all_keys = all_keys.drop_duplicates().sort_values(["db", "file"]).reset_index(drop=True)

    raw_by_key = pd.concat(raw_frames, ignore_index=True).dropna(subset=["raw_nfiq2"])
    raw_agg = raw_by_key.groupby(["file", "db"])["raw_nfiq2"].agg(["mean", "nunique", "std"])
    inconsistent = raw_agg[raw_agg["nunique"] > 1]
    inconsistency_notes = []
    if len(inconsistent) > 0:
        for (fname, dbname), row in inconsistent.iterrows():
            inconsistency_notes.append(
                f"{dbname}/{fname}: raw NFIQ2 disagrees across pipeline sources "
                f"(std={row['std']:.2f}) -- using the mean"
            )
    raw_lookup = raw_agg["mean"]

    master = all_keys.copy()
    master["Raw NFIQ2"] = master.apply(
        lambda r: raw_lookup.get((r["file"], r["db"])), axis=1
    )
    for key, label in PIPELINE_LABELS.items():
        series = enhanced.get(key)
        if series is None:
            master[label] = pd.NA
        else:
            master[label] = master.apply(
                lambda r, s=series: s.get((r["file"], r["db"])), axis=1
            )
    master["Hybrid"] = pd.NA  # never fabricated -- depends on Pipeline B finishing

    master = master.rename(columns={"file": "Image", "db": "DB"})
    master = master[["Image", "DB", "Raw NFIQ2", "Pipeline A", "Pipeline B",
                      "Pipeline C", "Pipeline D", "Hybrid"]]
    return master, inconsistency_notes


def build_summary_table(master):
    """Method (Raw/A/B/C/D/Hybrid) x DB1-4 Mean x Overall Mean x Δ vs Raw."""
    methods = ["Raw NFIQ2", "Pipeline A", "Pipeline B", "Pipeline C", "Pipeline D", "Hybrid"]
    dbs = sorted(master["DB"].dropna().unique())
    rows = []
    raw_overall_mean = master["Raw NFIQ2"].mean()
    for method in methods:
        if method not in master.columns:
            continue
        row = {"Method": method.replace(" NFIQ2", "") if method == "Raw NFIQ2" else method}
        for db in dbs:
            sub = master.loc[master["DB"] == db, method]
            row[f"{db} Mean"] = sub.mean() if sub.notna().any() else pd.NA
        overall = master[method]
        row["Overall Mean"] = overall.mean() if overall.notna().any() else pd.NA
        if method == "Raw NFIQ2" or pd.isna(row["Overall Mean"]):
            row["Δ vs Raw"] = pd.NA if method != "Raw NFIQ2" else 0.0
        else:
            row["Δ vs Raw"] = row["Overall Mean"] - raw_overall_mean
        rows.append(row)
    return pd.DataFrame(rows)


def build_improvement_stats(master):
    """% improved / degraded / unchanged per DB, per pipeline (vs Raw NFIQ2
    on the same image). Rows with a missing raw or pipeline score for that
    image are excluded from that pipeline's percentages rather than counted
    as any category, so a partially-populated pipeline still reports
    honest percentages over the images it actually has."""
    dbs = sorted(master["DB"].dropna().unique())
    rows = []
    for label in ("Pipeline A", "Pipeline B", "Pipeline C", "Pipeline D", "Hybrid"):
        if label not in master.columns:
            continue
        for db in dbs:
            sub = master.loc[master["DB"] == db, ["Raw NFIQ2", label]].dropna()
            n = len(sub)
            if n == 0:
                rows.append(dict(Pipeline=label, DB=db, n=0,
                                  **{"% improved": pd.NA, "% degraded": pd.NA, "% unchanged": pd.NA}))
                continue
            delta = sub[label] - sub["Raw NFIQ2"]
            improved = (delta > 0).sum()
            degraded = (delta < 0).sum()
            unchanged = (delta == 0).sum()
            rows.append(dict(
                Pipeline=label, DB=db, n=n,
                **{
                    "% improved": round(100 * improved / n, 1),
                    "% degraded": round(100 * degraded / n, 1),
                    "% unchanged": round(100 * unchanged / n, 1),
                },
            ))
    return pd.DataFrame(rows)
