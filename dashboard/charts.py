"""
Small chart helpers for the Overview/Comparison tab. Streamlit's built-in
st.bar_chart / st.altair_chart are enough here -- no heavy charting library
needed for a bar chart and a histogram.
"""

import pandas as pd
import streamlit as st


def mean_nfiq2_bar_chart(summary_df):
    """Bar chart of mean NFIQ2 per method, grouped by DB -- built from
    build_summary_table()'s output (Method x DB1..DB4 Mean x Overall Mean x
    Δ vs Raw). Only the per-DB mean columns are plotted; Overall/Δ stay in
    the table above the chart."""
    db_cols = [c for c in summary_df.columns if c.endswith(" Mean") and c != "Overall Mean"]
    if not db_cols or "Method" not in summary_df.columns:
        st.info("Not enough data yet to draw the per-DB bar chart.")
        return
    chart_df = summary_df.set_index("Method")[db_cols].rename(
        columns=lambda c: c.replace(" Mean", "")
    )
    chart_df = chart_df.dropna(how="all")
    if chart_df.empty:
        st.info("No pipeline has any scored images yet — nothing to chart.")
        return
    st.bar_chart(chart_df.T)  # DB on the x-axis, one bar per method


def before_after_distribution_chart(master_df, pipeline_label):
    """Overlaid histogram of Raw NFIQ2 vs. the selected pipeline's enhanced
    NFIQ2, across every scored image currently in the master table."""
    if pipeline_label not in master_df.columns:
        st.info(f"{pipeline_label} has no data loaded yet — nothing to chart.")
        return
    sub = master_df[["Raw NFIQ2", pipeline_label]].dropna()
    if sub.empty:
        st.info(f"No image has both a Raw and {pipeline_label} score yet — nothing to chart.")
        return

    long_df = pd.concat([
        pd.DataFrame({"NFIQ2": sub["Raw NFIQ2"], "Source": "Raw"}),
        pd.DataFrame({"NFIQ2": sub[pipeline_label], "Source": pipeline_label}),
    ], ignore_index=True)

    try:
        import altair as alt
        chart = (
            alt.Chart(long_df)
            .mark_bar(opacity=0.6)
            .encode(
                x=alt.X("NFIQ2:Q", bin=alt.Bin(maxbins=30), title="NFIQ2 score"),
                y=alt.Y("count()", stack=None, title="Number of images"),
                color=alt.Color("Source:N"),
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        # Fallback if altair isn't installed for some reason -- still shows
        # something useful rather than erroring the whole tab out.
        st.bar_chart(
            pd.DataFrame({
                "Raw": sub["Raw NFIQ2"],
                pipeline_label: sub[pipeline_label],
            }).reset_index(drop=True)
        )
