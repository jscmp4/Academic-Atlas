"""
Research Landscape Map — Interactive web application.

Workflow:
1. Enter Anthropic API key (stored in browser only)
2. Describe research interest → Claude intelligently searches local 3.4M paper DB
3. Click "Visualize" → embed + cluster → interactive scatter plot
4. Explore: click papers, AI cluster analysis, AI landscape overview
5. Export CSV / BibTeX / PNG
"""

import json
import os
import signal
import sqlite3
import subprocess
import threading
import time
import traceback
from pathlib import Path

import datamapplot
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
import dash
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update, ALL
from dash.exceptions import PreventUpdate

try:
    import dash_bootstrap_components as dbc
except ImportError:
    raise ImportError("pip install dash-bootstrap-components")

from utils import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

# Global progress tracker for ego network building
_ego_progress = {"percent": 0, "message": "", "running": False, "author_id": None,
                 "result_fig": None, "result_n": 0, "done": False}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def discover_datasets() -> dict[str, Path]:
    datasets = {}
    for f in sorted(DATA_DIR.glob("*_clustered.csv")):
        label = f.stem.replace("_clustered", "").replace("_", " ").title()
        if label.lower() == "papers":
            label = "Default (IS)"
        elif label.lower() == "worldmap":
            label = "World Map (all fields)"
        datasets[label] = f
    # Prioritize World Map as first option
    if "World Map (all fields)" in datasets:
        reordered = {"World Map (all fields)": datasets["World Map (all fields)"]}
        reordered.update({k: v for k, v in datasets.items() if k != "World Map (all fields)"})
        return reordered
    return datasets


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col, default in [
        ("x", 0.0), ("y", 0.0), ("cluster_id", -1),
        ("cluster_label", "Unknown"), ("topic_words", ""),
        ("citation_count", 0), ("year", 0), ("title", ""),
        ("abstract", ""), ("authors", ""), ("journal", ""),
        ("doi", ""), ("streams", ""), ("openalex_id", ""),
        ("journal_tier", ""), ("subfield", ""), ("field", ""),
    ]:
        if col not in df.columns:
            df[col] = default
    for col in ["cluster_label", "topic_words", "abstract", "authors", "doi", "streams", "journal_tier",
                "subfield", "field"]:
        df[col] = df[col].fillna("")
    return df


def load_user_papers() -> pd.DataFrame | None:
    path = DATA_DIR / "user_papers.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# BibTeX
# ---------------------------------------------------------------------------
def to_bibtex(df: pd.DataFrame) -> str:
    entries = []
    for i, row in df.iterrows():
        first_author = str(row.get("authors", "")).split(";")[0].strip()
        last_name = first_author.split()[-1] if first_author else "Unknown"
        year = int(row.get("year", 0))
        doi = str(row.get("doi", ""))
        entry = f"""@article{{{last_name}{year}_{i},
  title = {{{row.get('title', '')}}},
  author = {{{row.get('authors', '')}}},
  journal = {{{row.get('journal', '')}}},
  year = {{{year}}},
  doi = {{{doi}}},
}}"""
        entries.append(entry)
    return "\n\n".join(entries)


# ---------------------------------------------------------------------------
# Embed + Cluster search results into a visualizable dataset
# ---------------------------------------------------------------------------
def embed_and_cluster_papers(papers: list[dict]) -> pd.DataFrame:
    """Take raw search results, embed them, cluster, and return a ready-to-plot DataFrame."""
    from embed import generate_embeddings_from_texts
    from cluster import cluster_from_embeddings

    df = pd.DataFrame(papers)

    # Generate embeddings from title + abstract
    texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()
    embeddings = generate_embeddings_from_texts(texts)

    # Cluster
    df = cluster_from_embeddings(df, embeddings)

    return df


# ---------------------------------------------------------------------------
# Scatter figure
# ---------------------------------------------------------------------------
def build_scatter(df: pd.DataFrame, user_papers: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="No papers to display")
        return fig

    sizes = np.log1p(df["citation_count"].values)
    sizes = (sizes / max(sizes.max(), 1)) * 25 + 4

    clusters = df["cluster_label"].unique()
    palette = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    ]
    colors = {c: palette[i % len(palette)] for i, c in enumerate(sorted(clusters))}

    for cluster_label in sorted(clusters):
        mask = df["cluster_label"] == cluster_label
        cluster_df = df[mask]
        cluster_sizes = sizes[mask.values]
        fig.add_trace(go.Scatter(
            x=cluster_df["x"], y=cluster_df["y"], mode="markers",
            name=cluster_label[:40],
            marker=dict(size=cluster_sizes, color=colors[cluster_label],
                        opacity=0.7, line=dict(width=0.5, color="white")),
            text=cluster_df["title"], customdata=cluster_df.index.values,
            hovertemplate="<b>%{text}</b><br>Citations: %{marker.size:.0f}<extra>%{fullData.name}</extra>",
        ))

    if user_papers is not None and not user_papers.empty:
        user_dois = set(user_papers["doi"].dropna().str.lower()) if "doi" in user_papers.columns else set()
        user_titles = set(user_papers["title"].dropna().str.lower()) if "title" in user_papers.columns else set()
        star_mask = df["doi"].str.lower().isin(user_dois) | df["title"].str.lower().isin(user_titles)
        if star_mask.any():
            star_df = df[star_mask]
            fig.add_trace(go.Scatter(
                x=star_df["x"], y=star_df["y"], mode="markers", name="My Papers",
                marker=dict(size=18, color="gold", symbol="star", line=dict(width=2, color="black")),
                text=star_df["title"], customdata=star_df.index.values,
                hovertemplate="<b>%{text}</b><extra>My Paper</extra>",
            ))

    fig.update_layout(
        title="Research Landscape Map",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", height=700,
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(title="Clusters", font=dict(size=10), itemsizing="constant"),
        clickmode="event",
    )
    return fig


# ---------------------------------------------------------------------------
# Territory-style Paper Landscape  (cartographic map design)
# ---------------------------------------------------------------------------
# 5 pastel colours (top 4 fields + Other) — low saturation for dark bg.
_TERRITORY_PALETTE_RGB = [
    (126, 181, 227),   # soft blue      — Business, Mgmt & Accounting
    (240, 168, 144),   # soft salmon    — Social Sciences
    (142, 208, 158),   # soft green     — Computer Science
    (196, 168, 224),   # soft purple    — Decision Sciences
    (168, 168, 176),   # warm grey      — Other
]
_TERRITORY_PALETTE = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in _TERRITORY_PALETTE_RGB]

# Module-level cache: {data_path → background dict}
_territory_bg_cache: dict = {}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _assign_territories(
    df: pd.DataFrame, level: str = "field", max_n: int = 7,
) -> pd.Series:
    """Assign papers to territories.

    level="field"    → top 4 fields + Other  (overview / background)
    level="subfield" → top max_n subfields + Other  (detail labels)
    """
    col = "field" if level == "field" else "subfield"
    top_n = 4 if level == "field" else max_n
    vals = df[col].fillna("").replace("", pd.NA)
    if vals.notna().sum() < len(df) * 0.3:
        alt = "subfield" if col == "field" else "field"
        if alt in df.columns:
            vals = df[alt].fillna("").replace("", pd.NA)
        if vals.notna().sum() < len(df) * 0.3:
            from scipy.cluster.hierarchy import linkage, cut_tree
            k = min(top_n, max(3, len(df) // 50))
            Z = linkage(df[["x", "y"]].values, method="ward")
            labels = cut_tree(Z, n_clusters=[k])[:, 0]
            return pd.Series([f"Group {i}" for i in labels], index=df.index)
    counts = vals.dropna().value_counts()
    top_names = counts.head(top_n).index.tolist()
    territory = vals.where(vals.isin(top_names), other="Other")
    return territory.fillna("Other")


# ---- Pre-rendered KDE background + dominance-peak labels -------------------

def _render_territory_png(
    df: pd.DataFrame,
    territory: pd.Series,
    territory_names: list[str],
    territory_rgb: dict[str, tuple[int, int, int]],
    extent: tuple[float, float, float, float],
    grid_res: int = 200,
) -> tuple[str | None, dict[str, tuple[float, float]]]:
    """Render KDE territory background as PNG.

    Returns (base64_png_or_None, {territory_name: (label_x, label_y)}).
    Label positions are computed as "dominance peaks" — the point where
    each territory's density most exceeds all others.
    """
    import io
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from scipy.ndimage import gaussian_filter

    x_min, x_max, y_min, y_max = extent
    xgrid = np.linspace(x_min, x_max, grid_res)
    ygrid = np.linspace(y_min, y_max, grid_res)
    X, Y = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    density_grids: dict[str, np.ndarray] = {}
    rng = np.random.RandomState(42)
    for name in territory_names:
        t_df = df[territory == name]
        if len(t_df) < 10:
            continue
        pts = t_df[["x", "y"]].values.T
        if pts.shape[1] > 300:
            idx = rng.choice(pts.shape[1], 300, replace=False)
            pts = pts[:, idx]
        try:
            kde = gaussian_kde(pts, bw_method=0.2)
            density_grids[name] = kde(positions).reshape(X.shape)
        except (np.linalg.LinAlgError, ValueError):
            continue

    if not density_grids:
        return None, {}

    # RGBA image — winner-take-all with density-based alpha
    rgba = np.zeros((grid_res, grid_res, 4), dtype=np.float32)
    max_density = np.zeros((grid_res, grid_res), dtype=np.float32)

    for name in territory_names:
        Z = density_grids.get(name)
        if Z is None:
            continue
        z_max = Z.max()
        if z_max < 1e-12:
            continue
        Z_norm = Z / z_max
        mask = Z > max_density
        r, g, b = territory_rgb[name]
        rgba[:, :, 0] = np.where(mask, r / 255.0, rgba[:, :, 0])
        rgba[:, :, 1] = np.where(mask, g / 255.0, rgba[:, :, 1])
        rgba[:, :, 2] = np.where(mask, b / 255.0, rgba[:, :, 2])
        alpha = np.clip(Z_norm * 0.55, 0, 0.45)
        rgba[:, :, 3] = np.where(mask, alpha, rgba[:, :, 3])
        max_density = np.maximum(max_density, Z)

    # Ocean: suppress very low density
    if (max_density > 0).any():
        ocean_thresh = np.percentile(max_density[max_density > 0], 8)
        rgba[max_density < ocean_thresh] = 0

    # Smooth alpha for softer edges
    rgba[:, :, 3] = gaussian_filter(rgba[:, :, 3], sigma=2)

    # Render PNG
    dpi = 150
    fig_w = grid_res / dpi
    fig_mpl, ax = plt.subplots(figsize=(fig_w, fig_w), dpi=dpi)
    ax.imshow(rgba, origin="lower", extent=[x_min, x_max, y_min, y_max],
              aspect="auto", interpolation="bilinear")
    ax.axis("off")
    fig_mpl.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                    transparent=True, dpi=dpi)
    plt.close(fig_mpl)
    b64 = base64.b64encode(buf.getvalue()).decode()

    # Dominance peaks → label positions
    label_positions: dict[str, tuple[float, float]] = {}
    for name in territory_names:
        Z = density_grids.get(name)
        if Z is None:
            t_df = df[territory == name]
            if len(t_df) > 0:
                label_positions[name] = (float(t_df["x"].median()),
                                         float(t_df["y"].median()))
            continue
        # Dominance = own density − max(others)
        others_max = np.zeros_like(Z)
        for other in territory_names:
            if other != name and other in density_grids:
                others_max = np.maximum(others_max, density_grids[other])
        dominance = Z - others_max
        peak = np.unravel_index(dominance.argmax(), dominance.shape)
        label_positions[name] = (float(xgrid[peak[1]]), float(ygrid[peak[0]]))

    return b64, label_positions


def _get_territory_background(df: pd.DataFrame, data_path: str | None = None) -> dict:
    """Compute and cache territory background for a dataset."""
    cache_key = data_path or str(id(df))
    if cache_key in _territory_bg_cache:
        return _territory_bg_cache[cache_key]

    # Field-level (for background + overview labels + dot colours)
    field_terr = _assign_territories(df, level="field")
    field_names = [t for t in field_terr.value_counts().index if t != "Other"]
    if "Other" in field_terr.values:
        field_names.append("Other")
    field_colors: dict[str, str] = {}
    field_rgb: dict[str, tuple[int, int, int]] = {}
    for i, name in enumerate(field_names):
        field_colors[name] = _TERRITORY_PALETTE[i % len(_TERRITORY_PALETTE)]
        field_rgb[name] = _TERRITORY_PALETTE_RGB[i % len(_TERRITORY_PALETTE_RGB)]

    # Subfield-level (for detail labels)
    sub_terr = _assign_territories(df, level="subfield", max_n=7)
    sub_names = [t for t in sub_terr.value_counts().index if t != "Other"]
    if "Other" in sub_terr.values:
        sub_names.append("Other")

    # Extent
    pad = (df["x"].max() - df["x"].min()) * 0.08
    extent = (float(df["x"].min() - pad), float(df["x"].max() + pad),
              float(df["y"].min() - pad), float(df["y"].max() + pad))

    # PNG + coarse label positions (dominance peaks)
    bg_b64, coarse_pos = _render_territory_png(
        df, field_terr, field_names, field_rgb, extent)

    # Fine label positions (subfield median)
    fine_pos: dict[str, tuple[float, float]] = {}
    for name in sub_names:
        t_df = df[sub_terr == name]
        if len(t_df) >= 5:
            fine_pos[name] = (float(t_df["x"].median()), float(t_df["y"].median()))

    result = dict(bg_b64=bg_b64, field_terr=field_terr, sub_terr=sub_terr,
                  field_names=field_names, sub_names=sub_names,
                  field_colors=field_colors, field_rgb=field_rgb,
                  coarse_pos=coarse_pos, fine_pos=fine_pos, extent=extent)
    _territory_bg_cache[cache_key] = result
    return result


# ---- Main builder ----------------------------------------------------------

def build_territory_scatter(
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame | None = None,
    user_papers: pd.DataFrame | None = None,
    data_path: str | None = None,
    title: str = "Research Landscape Map",
) -> go.Figure:
    """Build cartographic Paper Landscape.

    full_df:     used for stable background + territory assignment
    filtered_df: used for scatter dots (if None, use full_df)
    """
    fig = go.Figure()
    plot_df = filtered_df if filtered_df is not None else full_df
    if plot_df.empty:
        fig.update_layout(title="No papers to display")
        return fig

    bg = _get_territory_background(full_df, data_path)
    extent = bg["extent"]

    # PNG background (0 polygon traces)
    if bg["bg_b64"]:
        fig.add_layout_image(
            source=f"data:image/png;base64,{bg['bg_b64']}",
            xref="x", yref="y",
            x=extent[0], y=extent[3],
            sizex=extent[1] - extent[0],
            sizey=extent[3] - extent[2],
            sizing="stretch", layer="below", opacity=1.0,
        )

    # Scattergl — small dots coloured by field
    field_terr = bg["field_terr"]
    plot_terr = field_terr.loc[plot_df.index]
    sizes = np.log1p(plot_df["citation_count"].values.astype(float))
    sizes = (sizes / max(sizes.max(), 1)) * 4 + 3
    point_colors = [bg["field_colors"].get(t, "#a8a8b0") for t in plot_terr]

    fig.add_trace(go.Scattergl(
        x=plot_df["x"], y=plot_df["y"], mode="markers",
        marker=dict(size=sizes, color=point_colors, opacity=0.35, line=dict(width=0)),
        text=plot_df["title"], customdata=plot_df.index.values,
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False,
    ))

    # User papers overlay
    if user_papers is not None and not user_papers.empty:
        user_dois = set(user_papers["doi"].dropna().str.lower()) if "doi" in user_papers.columns else set()
        user_titles = set(user_papers["title"].dropna().str.lower()) if "title" in user_papers.columns else set()
        star_mask = plot_df["doi"].str.lower().isin(user_dois) | plot_df["title"].str.lower().isin(user_titles)
        if star_mask.any():
            star_df = plot_df[star_mask]
            fig.add_trace(go.Scatter(
                x=star_df["x"], y=star_df["y"], mode="markers", name="My Papers",
                marker=dict(size=18, color="gold", symbol="star",
                            line=dict(width=2, color="black")),
                text=star_df["title"], customdata=star_df.index.values,
                hovertemplate="<b>%{text}</b><extra>My Paper</extra>",
            ))

    # ---- Annotations: coarse (field) + fine (subfield) ---------------------
    coarse_annotations = []
    for name in bg["field_names"]:
        pos = bg["coarse_pos"].get(name)
        if pos is None:
            continue
        count = int((plot_terr == name).sum())
        if count == 0:
            continue
        max_count = max(int((plot_terr == n).sum()) for n in bg["field_names"])
        font_size = max(13, min(18, int(13 + 5 * (count / max(max_count, 1)) ** 0.5)))
        coarse_annotations.append(dict(
            x=pos[0], y=pos[1],
            text=f"<b>{name}</b><br><i>({count:,})</i>",
            showarrow=False,
            font=dict(size=font_size, color="rgba(255,255,255,0.92)"),
            bgcolor="rgba(0,0,0,0.45)", borderpad=4, visible=True,
        ))
    n_coarse = len(coarse_annotations)

    fine_annotations = []
    sub_terr = bg["sub_terr"]
    plot_sub = sub_terr.loc[plot_df.index]
    for name in bg["sub_names"]:
        pos = bg["fine_pos"].get(name)
        if pos is None:
            continue
        count = int((plot_sub == name).sum())
        if count == 0:
            continue
        fine_annotations.append(dict(
            x=pos[0], y=pos[1],
            text=f"<b>{name}</b><br><i>({count:,})</i>",
            showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,0.85)"),
            bgcolor="rgba(0,0,0,0.35)", borderpad=3, visible=False,
        ))
    all_annotations = coarse_annotations + fine_annotations
    n_total = len(all_annotations)

    # Legend (one per field)
    for name in bg["field_names"]:
        count = int((plot_terr == name).sum())
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=bg["field_colors"].get(name, "#a8a8b0")),
            name=f"{name} ({count:,})",
        ))

    # Overview / Detail toggle buttons
    overview_args = {f"annotations[{i}].visible": i < n_coarse
                     for i in range(n_total)}
    detail_args = {f"annotations[{i}].visible": i >= n_coarse
                   for i in range(n_total)}

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="",
                   range=[extent[0], extent[1]]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="",
                   range=[extent[2], extent[3]]),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", height=700,
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(title="Research Fields", font=dict(size=10), itemsizing="constant"),
        clickmode="event",
        annotations=all_annotations,
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(30,30,60,0.85)",
            font=dict(color="white", size=11),
            pad=dict(r=8, t=4, b=4),
            buttons=[
                dict(label="🗺 Overview", method="relayout", args=[overview_args]),
                dict(label="🔍 Detail", method="relayout", args=[detail_args]),
            ],
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# DataMapPlot — interactive HTML map (replaces Plotly scatter for Paper Landscape)
# ---------------------------------------------------------------------------
_datamapplot_cache: dict = {}   # {cache_key: (html_path, timestamp)}


def _generate_datamapplot_html(df: pd.DataFrame, cache_key: str = "default") -> str:
    """Generate datamapplot interactive HTML, save to assets/, return iframe src path.

    Uses 3-layer labels: cluster_label (fine) → subfield (mid) → field (coarse).
    Results are cached by cache_key so identical filter combos don't regenerate.
    """
    if cache_key in _datamapplot_cache:
        path, ts = _datamapplot_cache[cache_key]
        if Path(path).exists():
            return f"/assets/{Path(path).name}?v={ts}"

    coords = df[["x", "y"]].values.astype(np.float32)

    # Label layers — finest first, coarsest last
    cluster_labels = df["cluster_label"].fillna("Unlabelled").values.astype(str)
    subfield_labels = df["subfield"].fillna("Other").values.astype(str)
    field_labels = df["field"].fillna("Other").values.astype(str)

    # Hover text
    hover = []
    for _, row in df.iterrows():
        title = str(row.get("title", "Untitled"))
        year = str(row.get("year", ""))
        journal = str(row.get("journal", "N/A"))
        cites = int(row.get("citation_count", 0))
        tier = str(row.get("journal_tier", ""))
        parts = [f"{title}", f"{year}  |  {journal}", f"Citations: {cites:,}"]
        if tier:
            parts.append(f"Tier: {tier}")
        hover.append("\n".join(parts))
    hover_text = np.array(hover)

    # Marker sizes — log-scaled by citations
    raw = np.log1p(df["citation_count"].values.astype(float))
    sizes = (raw / max(raw.max(), 1)) * 6 + 1.5

    ts = int(time.time())
    out_name = "paper_landscape.html"
    out_path = PROJECT_ROOT / "assets" / out_name

    plot = datamapplot.create_interactive_plot(
        coords,
        cluster_labels,
        subfield_labels,
        field_labels,
        hover_text=hover_text,
        darkmode=True,
        enable_search=True,
        enable_topic_tree=True,
        marker_size_array=sizes,
        inline_data=True,
        noise_label="Unlabelled",
        title="Research Landscape Map",
        color_label_text=True,
        cluster_boundary_polygons=False,
        point_radius_min_pixels=0.5,
        point_radius_max_pixels=16,
        min_fontsize=14,
        max_fontsize=28,
    )
    plot.save(str(out_path))

    # Inject time slider (CT scan mode) if year data is available
    if "year" in df.columns and df["year"].notna().any():
        _inject_time_slider(out_path, df["year"].fillna(0).astype(int).values)

    _datamapplot_cache[cache_key] = (str(out_path), ts)
    return f"/assets/{out_name}?v={ts}"


def _inject_time_slider(html_path: Path, years: np.ndarray):
    """Inject a time slider into datamapplot HTML that uses deck.gl DataFilterExtension.

    datamapplot already creates ScatterplotLayer with DataFilterExtension.
    We replace its getFilterValue (used for selection) with year-based filtering,
    and add a slider UI.
    """
    import json

    year_min = int(years.min())
    year_max = int(years.max())
    years_json = json.dumps(years.tolist())

    slider_html = f"""
<!-- Time Slider (CT Scan Mode) -->
<div id="time-slider-container" style="
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    z-index: 9999; background: rgba(0,0,0,0.85); padding: 12px 24px;
    border-radius: 8px; display: flex; align-items: center; gap: 12px;
    font-family: sans-serif; color: white; box-shadow: 0 2px 10px rgba(0,0,0,0.5);
">
    <button id="time-play-btn" style="
        background: #00cc96; border: none; color: white; padding: 4px 12px;
        border-radius: 4px; cursor: pointer; font-size: 14px;
    ">&#9654; Play</button>
    <span id="time-label" style="font-size: 14px; min-width: 120px;">
        {year_min} - {year_max}
    </span>
    <input type="range" id="time-slider" min="{year_min}" max="{year_max}"
        value="{year_max}" step="1" style="width: 400px; cursor: pointer;">
    <label style="font-size: 12px; opacity: 0.7;">
        <input type="checkbox" id="time-cumulative" checked> Cumulative
    </label>
</div>
<script>
(function() {{
    const years = {years_json};
    const yearMin = {year_min};
    const yearMax = {year_max};

    // Wait for datamap to be ready
    function waitForDatamap(cb) {{
        if (typeof datamap !== 'undefined' && datamap.pointLayer) cb();
        else setTimeout(() => waitForDatamap(cb), 500);
    }}

    waitForDatamap(function() {{
        const slider = document.getElementById('time-slider');
        const label = document.getElementById('time-label');
        const playBtn = document.getElementById('time-play-btn');
        const cumToggle = document.getElementById('time-cumulative');
        const n = years.length;

        // Create year attribute array as Float32Array
        const yearValues = new Float32Array(n);
        for (let i = 0; i < n; i++) yearValues[i] = years[i];

        function updateFilter(upTo) {{
            const cumulative = cumToggle.checked;
            const lo = cumulative ? yearMin : upTo;
            const hi = upTo;
            const softLo = cumulative ? yearMin - 1 : upTo - 2;

            label.textContent = cumulative
                ? yearMin + ' - ' + upTo
                : upTo.toString();

            datamap.updateTriggerCounter++;
            const updated = datamap.pointLayer.clone({{
                data: {{
                    ...datamap.pointLayer.props.data,
                    attributes: {{
                        ...datamap.pointLayer.props.data.attributes,
                        getFilterValue: {{ value: yearValues, size: 1 }}
                    }}
                }},
                filterRange: [lo, hi],
                filterSoftRange: [softLo, hi],
                updateTriggers: {{
                    getFilterValue: datamap.updateTriggerCounter
                }}
            }});
            const idx = datamap.layers.indexOf(datamap.pointLayer);
            datamap.layers = [
                ...datamap.layers.slice(0, idx),
                updated,
                ...datamap.layers.slice(idx + 1)
            ];
            datamap.deckgl.setProps({{ layers: datamap.layers }});
            datamap.pointLayer = updated;
        }}

        slider.addEventListener('input', function() {{
            updateFilter(parseInt(this.value));
        }});

        cumToggle.addEventListener('change', function() {{
            updateFilter(parseInt(slider.value));
        }});

        // Play animation
        let playing = false;
        let animFrame = null;
        playBtn.addEventListener('click', function() {{
            if (playing) {{
                playing = false;
                playBtn.innerHTML = '&#9654; Play';
                return;
            }}
            playing = true;
            playBtn.innerHTML = '&#9724; Stop';
            let year = yearMin;
            slider.value = year;

            function step() {{
                if (!playing || year > yearMax) {{
                    playing = false;
                    playBtn.innerHTML = '&#9654; Play';
                    return;
                }}
                slider.value = year;
                updateFilter(year);
                year++;
                animFrame = setTimeout(step, 300);
            }}
            step();
        }});
    }});
}})();
</script>
"""

    html = html_path.read_text(encoding="utf-8")
    # Insert before closing </body>
    html = html.replace("</body>", slider_html + "\n</body>")
    html_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Author scatter figure
# ---------------------------------------------------------------------------
def build_author_scatter(
    df: pd.DataFrame,
    highlight_name: str = "",
    coauthor_lines: list | None = None,
) -> go.Figure:
    """Build author landscape scatter plot."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="No authors to display")
        return fig

    sizes = np.log1p(df["total_citations"].values.astype(float))
    sizes = (sizes / max(sizes.max(), 1)) * 30 + 5

    clusters = df["cluster_label"].unique()
    palette = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    ]
    colors = {c: palette[i % len(palette)] for i, c in enumerate(sorted(clusters))}

    for cluster_label in sorted(clusters):
        mask = df["cluster_label"] == cluster_label
        cluster_df = df[mask]
        cluster_sizes = sizes[mask.values]

        hover_text = [
            f"<b>{row['name']}</b><br>"
            f"{row.get('institution', '')}<br>"
            f"{row['paper_count']} papers, {row['total_citations']:,} citations"
            for _, row in cluster_df.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=cluster_df["x"], y=cluster_df["y"], mode="markers",
            name=cluster_label[:40],
            marker=dict(size=cluster_sizes, color=colors[cluster_label],
                        opacity=0.7, line=dict(width=0.5, color="white")),
            text=hover_text, customdata=cluster_df.index.values,
            hovertemplate="%{text}<extra>%{fullData.name}</extra>",
        ))

    # Highlight searched author
    if highlight_name:
        hl_mask = df["name"].str.contains(highlight_name, case=False, na=False)
        if hl_mask.any():
            hl_df = df[hl_mask]
            fig.add_trace(go.Scatter(
                x=hl_df["x"], y=hl_df["y"], mode="markers+text",
                name="Highlighted",
                marker=dict(size=20, color="gold", symbol="star",
                            line=dict(width=2, color="black")),
                text=hl_df["name"], textposition="top center",
                textfont=dict(color="gold", size=11),
                customdata=hl_df.index.values,
                hovertemplate="<b>%{text}</b><extra>Highlighted</extra>",
            ))

    # Draw co-authorship lines
    if coauthor_lines:
        for x0, y0, x1, y1 in coauthor_lines:
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=1),
                showlegend=False, hoverinfo="skip",
            ))

    fig.update_layout(
        title="Author Landscape Map",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", height=700,
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(title="Research Groups", font=dict(size=10), itemsizing="constant"),
        clickmode="event",
    )
    return fig


def get_coauthor_lines(author_id: str, author_df: pd.DataFrame) -> list:
    """Get co-authorship lines for a given author. Returns [(x0,y0,x1,y1), ...]."""
    db_path = DATA_DIR / "openalex.db"
    conn = sqlite3.connect(str(db_path))

    # Find co-authors through shared papers
    coauthor_ids = conn.execute("""
        SELECT DISTINCT pa2.author_id
        FROM paper_authors pa1
        JOIN paper_authors pa2 ON pa1.paper_id = pa2.paper_id
        WHERE pa1.author_id = ? AND pa2.author_id != ?
    """, (author_id, author_id)).fetchall()

    conn.close()

    coauthor_set = {r[0] for r in coauthor_ids}

    # Get positions of the author and co-authors from the dataframe
    author_row = author_df[author_df["author_id"] == author_id]
    if author_row.empty:
        return []

    ax, ay = float(author_row.iloc[0]["x"]), float(author_row.iloc[0]["y"])

    lines = []
    for _, row in author_df.iterrows():
        if row["author_id"] in coauthor_set:
            lines.append((ax, ay, float(row["x"]), float(row["y"])))

    return lines


def get_author_details(author_id: str) -> dict:
    """Get author details: papers, co-authors."""
    db_path = DATA_DIR / "openalex.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Author info
    author = conn.execute(
        "SELECT * FROM authors WHERE author_id = ?", (author_id,)
    ).fetchone()

    # Papers
    papers = conn.execute("""
        SELECT p.title, p.year, p.journal, p.citation_count, p.doi, p.journal_tier
        FROM paper_authors pa
        JOIN papers p ON p.openalex_id = pa.paper_id
        WHERE pa.author_id = ?
        ORDER BY p.citation_count DESC
    """, (author_id,)).fetchall()

    # Top co-authors (by number of shared papers)
    coauthors = conn.execute("""
        SELECT a.name, a.institution, COUNT(*) as shared_papers, a.total_citations
        FROM paper_authors pa1
        JOIN paper_authors pa2 ON pa1.paper_id = pa2.paper_id AND pa1.author_id != pa2.author_id
        JOIN authors a ON a.author_id = pa2.author_id
        WHERE pa1.author_id = ?
        GROUP BY pa2.author_id
        ORDER BY shared_papers DESC
        LIMIT 15
    """, (author_id,)).fetchall()

    conn.close()

    return {
        "author": dict(author) if author else {},
        "papers": [dict(p) for p in papers],
        "coauthors": [dict(c) for c in coauthors],
    }


def build_ego_network(author_id: str) -> tuple[go.Figure, int]:
    """Build a 2-layer ego-network: focal → co-authors → co-authors' co-authors.

    Uses ALL papers per author for embedding (not a sample).
    Returns (figure, n_connections).
    """
    db_path = DATA_DIR / "openalex.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get focal author
    focal = conn.execute(
        "SELECT author_id, name, institution, paper_count, total_citations "
        "FROM authors WHERE author_id = ?", (author_id,)
    ).fetchone()
    if not focal:
        conn.close()
        return go.Figure(), 0

    def progress(pct, msg):
        _ego_progress["percent"] = pct
        _ego_progress["message"] = msg
        _ego_progress["running"] = True
        print(f"[EGO] {pct}% — {msg}")

    progress(5, f"Building network for {focal['name']}...")

    # Layer 1: direct co-authors
    progress(10, "Finding co-authors...")
    layer1 = conn.execute("""
        SELECT a.author_id, a.name, a.institution, a.paper_count, a.total_citations,
               COUNT(*) as shared_papers
        FROM paper_authors pa1
        JOIN paper_authors pa2 ON pa1.paper_id = pa2.paper_id AND pa1.author_id != pa2.author_id
        JOIN authors a ON a.author_id = pa2.author_id
        WHERE pa1.author_id = ?
        GROUP BY pa2.author_id
        ORDER BY shared_papers DESC
    """, (author_id,)).fetchall()

    layer1_ids = {c["author_id"] for c in layer1}

    # ----- Inner Circle Detection -----
    progress(12, "Detecting inner circle...")
    focal_paper_count = focal["paper_count"] or 1

    # Get paper years for each co-author relationship
    coauthor_years = conn.execute("""
        SELECT pa2.author_id, p.year
        FROM paper_authors pa1
        JOIN paper_authors pa2 ON pa1.paper_id = pa2.paper_id AND pa1.author_id != pa2.author_id
        JOIN papers p ON p.openalex_id = pa1.paper_id
        WHERE pa1.author_id = ?
    """, (author_id,)).fetchall()

    # Group years by co-author
    from collections import defaultdict
    coauthor_year_map = defaultdict(list)
    for row in coauthor_years:
        yr = row["year"]
        if yr:
            coauthor_year_map[row["author_id"]].append(yr)

    # Compute collaboration scores
    def compute_collaboration_score(shared_papers, ego_total, coauthor_total, paper_years, current_year=2026):
        freq = np.log1p(shared_papers)
        recency_scores = [0.85 ** (current_year - y) for y in paper_years if y]
        recency = np.mean(recency_scores) if recency_scores else 0
        jaccard = shared_papers / max(ego_total + coauthor_total - shared_papers, 1)
        return 0.4 * freq + 0.35 * recency + 0.25 * jaccard

    inner_circle_scores = {}
    for c in layer1:
        cid = c["author_id"]
        sp = c["shared_papers"]
        years = coauthor_year_map.get(cid, [])
        score = compute_collaboration_score(
            sp, focal_paper_count, c["paper_count"] or 1, years,
        )
        inner_circle_scores[cid] = {
            "score": score,
            "shared_papers": sp,
            "name": c["name"],
            "institution": c["institution"],
            "total_citations": c["total_citations"],
            "paper_years": years,
        }

    # Find gap cutoff: largest gap in sorted scores, min 2 shared papers
    eligible = {k: v for k, v in inner_circle_scores.items() if v["shared_papers"] >= 2}
    inner_circle_ids = set()
    if eligible:
        ranked = sorted(eligible.items(), key=lambda x: x[1]["score"], reverse=True)
        scores_sorted = [s[1]["score"] for s in ranked]
        if len(scores_sorted) >= 2:
            gaps = [scores_sorted[i] - scores_sorted[i+1] for i in range(len(scores_sorted)-1)]
            cutoff_idx = int(np.argmax(gaps))
            # Sanity: inner circle should be 2-15 people
            cutoff_idx = max(1, min(cutoff_idx, 14))
            inner_circle_ids = {ranked[i][0] for i in range(cutoff_idx + 1)}
        elif len(scores_sorted) == 1:
            inner_circle_ids = {ranked[0][0]}

    # Compute circle density (how many inner circle members co-authored with each other)
    circle_density = 0.0
    if len(inner_circle_ids) >= 2:
        ic_list = list(inner_circle_ids)
        ic_set_str = ",".join(f"'{a}'" for a in ic_list)
        # Count edges among inner circle members
        mutual_edges = conn.execute(f"""
            SELECT COUNT(DISTINCT pa1.author_id || '-' || pa2.author_id)
            FROM paper_authors pa1
            JOIN paper_authors pa2 ON pa1.paper_id = pa2.paper_id
            WHERE pa1.author_id IN ({ic_set_str})
              AND pa2.author_id IN ({ic_set_str})
              AND pa1.author_id < pa2.author_id
        """).fetchone()[0]
        possible = len(ic_list) * (len(ic_list) - 1) / 2
        circle_density = mutual_edges / possible if possible > 0 else 0

    # Get topic distribution for inner circle members
    ic_topics = []
    if inner_circle_ids:
        ic_str = ",".join(f"'{a}'" for a in inner_circle_ids)
        ic_topics_raw = conn.execute(f"""
            SELECT p.topic, COUNT(*) as cnt
            FROM paper_authors pa
            JOIN papers p ON p.openalex_id = pa.paper_id
            WHERE pa.author_id IN ({ic_str}) AND p.topic IS NOT NULL AND p.topic != ''
            GROUP BY p.topic ORDER BY cnt DESC LIMIT 10
        """).fetchall()
        ic_topics = [{"topic": r["topic"], "count": r["cnt"]} for r in ic_topics_raw]

    # Store for detail panel
    _ego_progress["inner_circle"] = {
        "ids": inner_circle_ids,
        "scores": {k: v for k, v in inner_circle_scores.items() if k in inner_circle_ids},
        "density": circle_density,
        "topics": ic_topics[:5],
        "n_members": len(inner_circle_ids),
    }

    progress(15, f"Found {len(layer1)} co-authors, {len(inner_circle_ids)} inner circle. Expanding to layer 2...")
    layer2_map = {}  # author_id -> {info + connections to layer1}
    for c in layer1:
        l2 = conn.execute("""
            SELECT a.author_id, a.name, a.institution, a.paper_count, a.total_citations,
                   COUNT(*) as shared_papers
            FROM paper_authors pa1
            JOIN paper_authors pa2 ON pa1.paper_id = pa2.paper_id AND pa1.author_id != pa2.author_id
            JOIN authors a ON a.author_id = pa2.author_id
            WHERE pa1.author_id = ? AND pa2.author_id != ?
            GROUP BY pa2.author_id
            HAVING shared_papers >= 2
        """, (c["author_id"], author_id)).fetchall()

        for a in l2:
            aid = a["author_id"]
            if aid in layer1_ids or aid == author_id:
                continue
            if aid not in layer2_map:
                layer2_map[aid] = {
                    "author_id": aid, "name": a["name"],
                    "institution": a["institution"],
                    "paper_count": a["paper_count"],
                    "total_citations": a["total_citations"],
                    "links_to_layer1": [],
                }
            layer2_map[aid]["links_to_layer1"].append(c["author_id"])

    # Keep layer2 authors connected to >= 2 layer1 people (to avoid explosion)
    layer2 = [v for v in layer2_map.values() if len(v["links_to_layer1"]) >= 2]
    layer2.sort(key=lambda x: x["total_citations"], reverse=True)
    layer2 = layer2[:100]  # Cap at 100 for performance
    layer2_ids = {a["author_id"] for a in layer2}
    progress(25, f"Layer 2: {len(layer2)} extended authors")

    # All authors in network
    all_ids = [author_id] + [c["author_id"] for c in layer1] + [a["author_id"] for a in layer2]
    all_info = {author_id: dict(focal)}
    for c in layer1:
        all_info[c["author_id"]] = dict(c)
    for a in layer2:
        all_info[a["author_id"]] = a

    # Get ALL papers for each author → build embedding text
    progress(30, f"Loading papers for {len(all_ids)} authors...")
    author_texts = {}
    for i, aid in enumerate(all_ids):
        papers = conn.execute("""
            SELECT p.title, p.abstract, p.citation_count
            FROM paper_authors pa JOIN papers p ON p.openalex_id = pa.paper_id
            WHERE pa.author_id = ?
            ORDER BY p.citation_count DESC
        """, (aid,)).fetchall()

        if papers:
            # Citation-weighted: repeat high-citation paper titles for emphasis
            parts = []
            for p in papers:
                title = p["title"] or ""
                abstract = (p["abstract"] or "")[:300]
                parts.append(f"{title}. {abstract}")
            author_texts[aid] = " ".join(parts)

        if (i + 1) % 20 == 0:
            pct = 30 + int(30 * (i + 1) / len(all_ids))
            progress(pct, f"Loading papers: {i+1}/{len(all_ids)} authors...")

    conn.close()

    if len(author_texts) < 3:
        return go.Figure(), 0

    # Embed all authors
    ordered_ids = [aid for aid in all_ids if aid in author_texts]
    texts = [author_texts[aid] for aid in ordered_ids]
    progress(65, f"Embedding {len(texts)} authors...")
    from embed import generate_embeddings_from_texts
    embeddings = generate_embeddings_from_texts(texts)

    # UMAP 2D
    n = len(ordered_ids)
    progress(80, f"UMAP projection ({n} points)...")
    if n >= 5:
        try:
            from umap import UMAP
            coords = UMAP(
                n_neighbors=min(15, n - 1), n_components=2,
                min_dist=0.3, metric="cosine", random_state=42,
            ).fit_transform(embeddings)
        except ImportError:
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)

    pos = {aid: (float(coords[i, 0]), float(coords[i, 1])) for i, aid in enumerate(ordered_ids)}

    # Build edges
    progress(90, "Building visualization...")
    fig = go.Figure()

    # Edges: focal → layer1 (gold for inner circle, blue for others)
    for c in layer1:
        cid = c["author_id"]
        if cid in pos and author_id in pos:
            x0, y0 = pos[author_id]
            x1, y1 = pos[cid]
            is_ic = cid in inner_circle_ids
            w = min(c["shared_papers"], 8)
            edge_color = "rgba(255,215,0,0.5)" if is_ic else "rgba(99,110,250,0.4)"
            edge_w = max(w, 3) if is_ic else w
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                line=dict(color=edge_color, width=edge_w),
                showlegend=False, hoverinfo="skip",
            ))

    # Edges: inner circle ↔ inner circle (white bold lines)
    ic_in_pos = [cid for cid in inner_circle_ids if cid in pos]
    if len(ic_in_pos) >= 2:
        # Check mutual co-authorship among inner circle
        ic_str = ",".join(f"'{a}'" for a in ic_in_pos)
        db_path = DATA_DIR / "openalex.db"
        conn2 = sqlite3.connect(str(db_path))
        ic_edges = conn2.execute(f"""
            SELECT DISTINCT pa1.author_id as a1, pa2.author_id as a2
            FROM paper_authors pa1
            JOIN paper_authors pa2 ON pa1.paper_id = pa2.paper_id
            WHERE pa1.author_id IN ({ic_str})
              AND pa2.author_id IN ({ic_str})
              AND pa1.author_id < pa2.author_id
        """).fetchall()
        conn2.close()
        for edge in ic_edges:
            a1, a2 = edge[0], edge[1]
            if a1 in pos and a2 in pos:
                x0, y0 = pos[a1]
                x1, y1 = pos[a2]
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                    line=dict(color="rgba(255,255,255,0.5)", width=3),
                    showlegend=False, hoverinfo="skip",
                ))

    # Edges: layer2 → layer1
    for a in layer2:
        aid = a["author_id"]
        if aid not in pos:
            continue
        for l1_id in a["links_to_layer1"]:
            if l1_id in pos:
                x0, y0 = pos[aid]
                x1, y1 = pos[l1_id]
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                    line=dict(color="rgba(255,255,255,0.12)", width=1),
                    showlegend=False, hoverinfo="skip",
                ))

    # Layer 2 dots (small, faded)
    if layer2:
        l2_x, l2_y, l2_text, l2_sizes = [], [], [], []
        for a in layer2:
            if a["author_id"] in pos:
                l2_x.append(pos[a["author_id"]][0])
                l2_y.append(pos[a["author_id"]][1])
                l2_text.append(
                    f"<b>{a['name']}</b><br>{a.get('institution', '') or ''}<br>"
                    f"{a['total_citations']:,} cites"
                )
                l2_sizes.append(max(4, min(15, np.log1p(a["total_citations"]) * 1.5)))
        fig.add_trace(go.Scatter(
            x=l2_x, y=l2_y, mode="markers",
            name="Extended network",
            marker=dict(size=l2_sizes, color="#7F7F7F", opacity=0.5,
                        line=dict(width=0.3, color="white")),
            text=l2_text, hovertemplate="%{text}<extra></extra>",
        ))

    # Layer 1 dots — split into inner circle (gold border, larger, with name labels)
    # and periphery (regular blue)
    ic_x, ic_y, ic_text, ic_sizes, ic_names = [], [], [], [], []
    per_x, per_y, per_text, per_sizes = [], [], [], []
    for c in layer1:
        cid = c["author_id"]
        if cid not in pos:
            continue
        is_ic = cid in inner_circle_ids
        hover = (
            f"<b>{c['name']}</b><br>{c['institution'] or ''}<br>"
            f"{c['shared_papers']} shared papers, {c['total_citations']:,} cites"
        )
        if is_ic:
            ic_x.append(pos[cid][0])
            ic_y.append(pos[cid][1])
            ic_text.append(hover)
            ic_sizes.append(max(10, min(28, np.log1p(c["total_citations"]) * 2.5)))
            ic_names.append(c["name"].split()[-1])  # last name for label
        else:
            per_x.append(pos[cid][0])
            per_y.append(pos[cid][1])
            per_text.append(hover)
            per_sizes.append(max(6, min(25, np.log1p(c["total_citations"]) * 2.5)))

    if per_x:
        fig.add_trace(go.Scatter(
            x=per_x, y=per_y, mode="markers",
            name="Co-authors",
            marker=dict(size=per_sizes, color="#636EFA", opacity=0.85,
                        line=dict(width=0.5, color="white")),
            text=per_text, hovertemplate="%{text}<extra></extra>",
        ))
    if ic_x:
        fig.add_trace(go.Scatter(
            x=ic_x, y=ic_y, mode="markers+text",
            name="Inner Circle",
            marker=dict(size=ic_sizes, color="#636EFA", opacity=0.95,
                        line=dict(width=2.5, color="gold")),
            text=ic_names,
            textposition="top center",
            textfont=dict(color="gold", size=10),
            hovertext=ic_text,
            hovertemplate="%{hovertext}<extra>Inner Circle</extra>",
        ))

    # Focal author (gold star)
    if author_id in pos:
        fx, fy = pos[author_id]
        fig.add_trace(go.Scatter(
            x=[fx], y=[fy], mode="markers+text",
            name=focal["name"],
            marker=dict(size=24, color="gold", symbol="star",
                        line=dict(width=2, color="black")),
            text=[focal["name"]], textposition="top center",
            textfont=dict(color="gold", size=12),
            hovertemplate=f"<b>{focal['name']}</b><br>{focal['institution']}<br>"
                          f"{focal['paper_count']} papers, {focal['total_citations']:,} cites<extra></extra>",
        ))

    n_total = len(layer1) + len(layer2)
    fig.update_layout(
        title=f"Research Network: {focal['name']} ({len(layer1)} co-authors, {len(layer2)} extended)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", height=700,
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(font=dict(size=10)),
        clickmode="event",
    )

    progress(95, "Building paper territory map...")

    # Also build paper-level territory map
    territory_fig = _build_paper_territory(author_id, focal, layer1, conn=None)
    _ego_progress["result_territory"] = territory_fig

    progress(100, "Done!")
    _ego_progress["running"] = False
    return fig, n_total


# ---------------------------------------------------------------------------
# Hierarchical cluster bubbles for Research Territory
# ---------------------------------------------------------------------------
def _compute_hierarchical_clusters(
    coords: np.ndarray, texts: list[str], papers: list[dict] | None = None,
) -> dict:
    """Compute 2-level hierarchical clusters on 2D UMAP coords.

    Uses scipy Ward linkage + cut_tree (single dendrogram, guaranteed nesting).
    Labels: OpenAlex topic/subfield if papers provided, TF-IDF fallback otherwise.
    """
    from scipy.cluster.hierarchy import linkage, cut_tree
    from sklearn.feature_extraction.text import TfidfVectorizer
    from collections import Counter

    n = len(coords)
    if n < 5:
        return None

    k_coarse = max(3, min(8, int(n ** 0.33)))
    k_fine = max(8, min(30, int(n ** 0.5 / 2)))
    k_fine = max(k_fine, k_coarse + 2)

    Z = linkage(coords, method='ward')
    all_labels = cut_tree(Z, n_clusters=[k_coarse, k_fine])
    coarse_labels = all_labels[:, 0].ravel()
    fine_labels = all_labels[:, 1].ravel()

    # TF-IDF as fallback
    vectorizer = TfidfVectorizer(max_features=8000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    feat_names = vectorizer.get_feature_names_out()

    def tfidf_label(indices, top_n):
        if len(indices) == 0:
            return "Unknown"
        mean_vec = tfidf[indices].mean(axis=0).A1
        top_idx = mean_vec.argsort()[-top_n:][::-1]
        return ", ".join(feat_names[i] for i in top_idx)

    def openalex_label(indices, field="topic"):
        """Get label from OpenAlex topic/subfield majority vote."""
        if not papers:
            return None
        values = [papers[i].get(field, "") for i in indices if papers[i].get(field)]
        if not values:
            return None
        counts = Counter(values)
        top_val, top_count = counts.most_common(1)[0]
        if top_count / len(values) > 0.5:
            return top_val
        top2 = counts.most_common(2)
        if len(top2) > 1:
            return f"{top2[0][0]} / {top2[1][0]}"
        return top_val

    def get_label(indices, level="fine"):
        """Get best available label: OpenAlex > TF-IDF."""
        if level == "coarse":
            label = openalex_label(indices, field="subfield")
            if label:
                return label
            return tfidf_label(indices, 3)
        else:
            label = openalex_label(indices, field="topic")
            if label:
                return label
            return tfidf_label(indices, 4)

    def make_ellipse(pts_2d, scale, n_pts=60):
        """Compute confidence ellipse. Returns (x_arr, y_arr) or None if < 3 pts."""
        cx, cy = float(pts_2d[:, 0].mean()), float(pts_2d[:, 1].mean())
        if len(pts_2d) < 3:
            return None, (cx, cy)
        cov = np.cov(pts_2d.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        a = scale * np.sqrt(eigenvalues[0])
        b = scale * np.sqrt(eigenvalues[1])
        theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        t = np.linspace(0, 2 * np.pi, n_pts)
        x = cx + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
        y = cy + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
        return (x, y), (cx, cy)

    bubble_palette = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880",
    ]

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    # Coarse clusters
    coarse_info = {}
    for cid in range(k_coarse):
        mask = coarse_labels == cid
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        pts = coords[mask]
        ellipse, centroid = make_ellipse(pts, scale=2.0)
        color = bubble_palette[cid % len(bubble_palette)]
        coarse_info[cid] = {
            "label": get_label(indices.tolist(), "coarse"),
            "centroid": centroid,
            "ellipse": ellipse,
            "color": color,
            "rgb": hex_to_rgb(color),
            "n_papers": int(len(indices)),
        }

    # Fine clusters + parent map
    fine_info = {}
    for fid in range(k_fine):
        mask = fine_labels == fid
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        pts = coords[mask]
        # Parent = the coarse cluster these points belong to (guaranteed unique by Ward)
        parent_id = int(coarse_labels[indices[0]])
        parent_color = coarse_info.get(parent_id, {}).get("color", "#636EFA")
        parent_rgb = coarse_info.get(parent_id, {}).get("rgb", (99, 110, 250))

        ellipse, centroid = make_ellipse(pts, scale=1.5)
        # Slightly lighter version of parent color
        r, g, b = parent_rgb
        lr = min(255, r + 40)
        lg = min(255, g + 40)
        lb = min(255, b + 40)

        fine_info[fid] = {
            "label": get_label(indices.tolist(), "fine"),
            "centroid": centroid,
            "ellipse": ellipse,
            "color": f"#{lr:02x}{lg:02x}{lb:02x}",
            "rgb": (lr, lg, lb),
            "parent_id": parent_id,
            "n_papers": int(len(indices)),
        }

    return {
        "coarse_info": coarse_info,
        "fine_info": fine_info,
        "coarse_labels": coarse_labels,
        "fine_labels": fine_labels,
    }


def _add_cluster_bubbles_to_fig(fig: go.Figure, cluster_info: dict):
    """Add hierarchical cluster ellipses + labels to a Plotly figure.

    Adds traces at the beginning so they render behind scatter dots.
    """
    coarse_info = cluster_info["coarse_info"]
    fine_info = cluster_info["fine_info"]

    # Coarse ellipses (bottom layer — dashed, very transparent)
    for cid in sorted(coarse_info):
        info = coarse_info[cid]
        ellipse = info["ellipse"]
        if ellipse is None:
            continue
        x_ell, y_ell = ellipse
        r, g, b = info["rgb"]
        fig.add_trace(go.Scatter(
            x=np.append(x_ell, x_ell[0]),
            y=np.append(y_ell, y_ell[0]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.06)",
            line=dict(color=f"rgba({r},{g},{b},0.3)", width=2, dash="dot"),
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
        ))

    # Count fine children per coarse parent — skip "only child" fine ellipses
    from collections import Counter
    children_per_parent = Counter(info["parent_id"] for info in fine_info.values())

    # Fine ellipses (middle layer — solid, slightly more opaque)
    for fid in sorted(fine_info):
        info = fine_info[fid]
        # Skip if this fine cluster is the only child of its parent (would overlap)
        if children_per_parent.get(info["parent_id"], 0) <= 1:
            continue
        ellipse = info["ellipse"]
        if ellipse is None:
            continue
        x_ell, y_ell = ellipse
        r, g, b = info["rgb"]
        fig.add_trace(go.Scatter(
            x=np.append(x_ell, x_ell[0]),
            y=np.append(y_ell, y_ell[0]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.10)",
            line=dict(color=f"rgba({r},{g},{b},0.4)", width=1),
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
        ))

    # Annotations: coarse labels (visible) + fine labels (hidden by default)
    annotations = []
    for cid in sorted(coarse_info):
        info = coarse_info[cid]
        cx, cy = info["centroid"]
        annotations.append(dict(
            x=cx, y=cy,
            text=f"<b>{info['label']}</b>",
            font=dict(color=info["color"], size=13),
            showarrow=False,
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=3,
        ))

    for fid in sorted(fine_info):
        info = fine_info[fid]
        # Skip label for only-child fine clusters (same position as coarse label)
        if children_per_parent.get(info["parent_id"], 0) <= 1:
            continue
        cx, cy = info["centroid"]
        annotations.append(dict(
            x=cx, y=cy,
            text=info["label"],
            font=dict(color="rgba(255,255,255,0.7)", size=9),
            showarrow=False,
            visible=False,  # hidden by default, toggle via switch
        ))

    fig.update_layout(annotations=annotations)


def _build_paper_territory(author_id: str, focal, layer1, conn=None):
    """Build paper-level scatter: each dot = a paper, colored by author."""
    db_path = DATA_DIR / "openalex.db"
    own_conn = False
    if conn is None:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        own_conn = True

    # Get focal author's papers
    focal_papers = conn.execute("""
        SELECT p.openalex_id, p.title, p.abstract, p.year, p.citation_count,
               p.journal, p.journal_tier, p.topic, p.subfield, p.field
        FROM paper_authors pa JOIN papers p ON p.openalex_id = pa.paper_id
        WHERE pa.author_id = ?
    """, (author_id,)).fetchall()

    # Get top co-authors' papers (limit to top 15 co-authors by shared papers)
    top_coauthors = list(layer1)[:15]

    coauthor_papers = {}  # author_name -> [papers]
    for ca in top_coauthors:
        papers = conn.execute("""
            SELECT p.openalex_id, p.title, p.abstract, p.year, p.citation_count,
                   p.journal, p.journal_tier
            FROM paper_authors pa JOIN papers p ON p.openalex_id = pa.paper_id
            WHERE pa.author_id = ?
        """, (ca["author_id"],)).fetchall()
        if papers:
            coauthor_papers[ca["name"]] = [dict(p) for p in papers]

    if own_conn:
        conn.close()

    # Collect all papers + labels
    all_papers = []
    labels = []

    for p in focal_papers:
        all_papers.append(dict(p))
        labels.append(focal["name"])

    for ca_name, papers in coauthor_papers.items():
        for p in papers:
            all_papers.append(p)
            labels.append(ca_name)

    if len(all_papers) < 5:
        return go.Figure()

    # Deduplicate by openalex_id (keep first occurrence → focal author takes priority)
    seen = set()
    unique_papers = []
    unique_labels = []
    for p, lbl in zip(all_papers, labels):
        pid = p["openalex_id"]
        if pid not in seen:
            seen.add(pid)
            unique_papers.append(p)
            unique_labels.append(lbl)

    # Embed
    from embed import generate_embeddings_from_texts
    texts = [f"{p['title']}. {(p['abstract'] or '')[:300]}" for p in unique_papers]
    embeddings = generate_embeddings_from_texts(texts)

    # UMAP 2D
    n = len(unique_papers)
    try:
        from umap import UMAP
        coords = UMAP(
            n_neighbors=min(15, n - 1), n_components=2,
            min_dist=0.1, metric="cosine", random_state=42,
        ).fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)

    # Hierarchical cluster bubbles
    cluster_info = _compute_hierarchical_clusters(coords, texts, unique_papers)

    # Build figure — bubbles first (bottom), then author dots (top)
    fig = go.Figure()
    if cluster_info:
        _add_cluster_bubbles_to_fig(fig, cluster_info)

    palette = [
        "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
        "#FF6692", "#B6E880", "#FF97FF", "#FECB52", "#636EFA",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    ]

    # Group by author
    author_names = list(dict.fromkeys(unique_labels))  # preserve order, focal first
    color_map = {}
    color_map[focal["name"]] = "gold"
    for i, name in enumerate(author_names):
        if name != focal["name"]:
            color_map[name] = palette[i % len(palette)]

    for author_name in author_names:
        mask = [i for i, lbl in enumerate(unique_labels) if lbl == author_name]
        if not mask:
            continue

        ax = [float(coords[i, 0]) for i in mask]
        ay = [float(coords[i, 1]) for i in mask]
        sizes = [max(5, min(20, np.log1p(unique_papers[i]["citation_count"]) * 2.5)) for i in mask]
        hover = [
            f"<b>{unique_papers[i]['title'][:60]}</b><br>"
            f"{unique_papers[i]['journal']} ({unique_papers[i]['year']})<br>"
            f"{unique_papers[i]['citation_count']:,} citations"
            for i in mask
        ]

        is_focal = (author_name == focal["name"])
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="markers",
            name=f"{'★ ' if is_focal else ''}{author_name[:30]} ({len(mask)})",
            marker=dict(
                size=sizes,
                color=color_map[author_name],
                opacity=0.9 if is_focal else 0.6,
                symbol="star" if is_focal else "circle",
                line=dict(width=1 if is_focal else 0.3, color="black" if is_focal else "white"),
            ),
            text=hover,
            hovertemplate="%{text}<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title=f"Research Territory: {focal['name']} & co-authors ({len(unique_papers)} papers)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", height=700,
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(title="Authors", font=dict(size=9), itemsizing="constant"),
        clickmode="event",
    )

    # Also build 3D trajectory using same data (pass cluster info for domain coloring)
    trajectory_fig = _build_3d_trajectory(
        author_id, focal, unique_papers, coords, unique_labels, cluster_info,
    )
    _ego_progress["result_3d"] = trajectory_fig

    return fig


# ---------------------------------------------------------------------------
# 3D Scholar Temporal Trajectory
# ---------------------------------------------------------------------------
def _build_3d_trajectory(
    author_id: str, focal, unique_papers, coords, unique_labels,
    cluster_info: dict | None = None,
) -> go.Figure:
    """Build 3D scatter: X/Y = UMAP, Z = year.

    Two coloring modes (toggle via buttons):
    - Year mode: Plasma colorscale by publication year
    - Domain mode: color each trajectory segment by nearest 2D cluster
    Also renders 3D ellipsoid bubbles for coarse clusters.
    """
    from scipy.spatial.distance import cdist

    if len(unique_papers) < 3:
        return go.Figure()

    # ----- Shared data -----
    all_x = [float(coords[i, 0]) for i in range(len(unique_papers))]
    all_y = [float(coords[i, 1]) for i in range(len(unique_papers))]
    all_z = [unique_papers[i].get("year", 2000) for i in range(len(unique_papers))]
    z_min = min(all_z) if all_z else 2000
    z_max = max(all_z) if all_z else 2026
    z_range = [z_min - 1, z_max + 1]

    focal_name = focal["name"]
    focal_indices = [i for i, lbl in enumerate(unique_labels) if lbl == focal_name]

    # ----- Assign each paper to nearest coarse cluster -----
    paper_cluster_ids = np.full(len(unique_papers), -1, dtype=int)
    coarse_info = cluster_info.get("coarse_info", {}) if cluster_info else {}
    coarse_labels_arr = cluster_info.get("coarse_labels") if cluster_info else None

    if coarse_labels_arr is not None:
        paper_cluster_ids = coarse_labels_arr.copy()
    elif coarse_info:
        # Fallback: nearest centroid assignment
        centroids = np.array([coarse_info[cid]["centroid"] for cid in sorted(coarse_info)])
        cid_list = sorted(coarse_info.keys())
        dists = cdist(coords[:, :2], centroids)
        nearest = dists.argmin(axis=1)
        paper_cluster_ids = np.array([cid_list[n] for n in nearest])

    bubble_palette = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880",
    ]

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def cluster_color(cid):
        if cid in coarse_info:
            return coarse_info[cid]["color"]
        return bubble_palette[int(cid) % len(bubble_palette)]

    # ===== Build focal author's paper dots: colored by cluster, sized by citation =====
    fig = go.Figure()

    # ----- Background: co-author papers as faint cluster-colored dots -----
    # (not focal author's papers — those get prominent treatment below)
    non_focal = [i for i in range(len(unique_papers)) if unique_labels[i] != focal_name]
    if non_focal:
        nf_x = [all_x[i] for i in non_focal]
        nf_y = [all_y[i] for i in non_focal]
        nf_z = [all_z[i] for i in non_focal]
        nf_colors = [cluster_color(int(paper_cluster_ids[i])) for i in non_focal]
        nf_sizes = [max(2, min(6, np.log1p(unique_papers[i].get("citation_count", 0)) * 0.8))
                     for i in non_focal]
        fig.add_trace(go.Scatter3d(
            x=nf_x, y=nf_y, z=nf_z,
            mode="markers",
            marker=dict(size=nf_sizes, color=nf_colors, opacity=0.12),
            showlegend=False, hoverinfo="skip",
        ))

    # ----- Focal author papers + branching phylogenetic tree -----
    if focal_indices:
        focal_data = [(
            float(coords[i, 0]), float(coords[i, 1]),
            unique_papers[i].get("year", 2000),
            unique_papers[i].get("title", "")[:60],
            unique_papers[i].get("citation_count", 0),
            unique_papers[i].get("journal", ""),
            int(paper_cluster_ids[i]),
        ) for i in focal_indices]
        focal_data.sort(key=lambda x: (x[2], -x[4]))  # sort by year, then citation desc

        from collections import defaultdict

        # Group focal papers by cluster
        cluster_groups = defaultdict(list)
        for d in focal_data:
            cluster_groups[d[6]].append(d)

        # ---- Branching tree: connect actual papers, fork at real positions ----
        #
        # Strategy per cluster:
        #   1. Sort papers by year
        #   2. Within same year, pick highest-citation as "spine node"
        #   3. Connect spine nodes sequentially (the branch backbone)
        #   4. Same-year non-spine papers connect to their spine node (short spokes)
        #
        # Fork point: when a new cluster first appears, it forks from the
        # last paper of ANY cluster before that year (the "global timeline")

        # Build global timeline: all focal papers sorted by year, pick the
        # last paper before each cluster's first appearance as fork origin
        global_timeline = sorted(focal_data, key=lambda d: (d[2], -d[4]))

        # Find fork origins: for each cluster, the most recent paper (any cluster)
        # published before this cluster's first paper
        cluster_first_year = {}
        for cid, papers in cluster_groups.items():
            cluster_first_year[cid] = min(d[2] for d in papers)

        fork_origins = {}  # cid -> (x, y, z) of the fork point
        for cid in cluster_groups:
            first_yr = cluster_first_year[cid]
            # Find the latest paper before this cluster started
            candidates = [d for d in global_timeline if d[2] < first_yr]
            if candidates:
                origin = candidates[-1]  # most recent paper before fork
                fork_origins[cid] = (origin[0], origin[1], origin[2])

        legend_seen = set()

        for cid, papers in cluster_groups.items():
            color = cluster_color(cid)
            clabel = coarse_info.get(cid, {}).get("label", f"Cluster {cid}") if coarse_info else f"Cluster {cid}"
            r, g, b = hex_to_rgb(color)

            # Sort by year, then citation desc
            papers_sorted = sorted(papers, key=lambda d: (d[2], -d[4]))

            # Group by year → pick spine nodes (highest citation per year)
            year_groups = defaultdict(list)
            for d in papers_sorted:
                year_groups[d[2]].append(d)

            spine_nodes = []  # main branch path (one per year)
            spoke_pairs = []  # (spine_node, satellite_node) for same-year papers
            for yr in sorted(year_groups.keys()):
                group = year_groups[yr]
                # Spine = highest citation in that year
                spine = group[0]  # already sorted by -citation
                spine_nodes.append(spine)
                for sat in group[1:]:
                    spoke_pairs.append((spine, sat))

            # ---- Draw branch backbone: fork_origin → spine_node1 → spine_node2 → ... ----
            bx, by, bz = [], [], []
            if cid in fork_origins:
                ox, oy, oz = fork_origins[cid]
                bx.append(ox)
                by.append(oy)
                bz.append(oz)
            for s in spine_nodes:
                bx.append(s[0])
                by.append(s[1])
                bz.append(s[2])

            if len(bx) >= 2:
                fig.add_trace(go.Scatter3d(
                    x=bx, y=by, z=bz,
                    mode="lines",
                    line=dict(color=f"rgba({r},{g},{b},0.55)", width=3.5),
                    showlegend=False, hoverinfo="skip",
                    legendgroup=clabel,
                ))

            # ---- Draw spokes: spine → satellite for same-year papers ----
            for spine, sat in spoke_pairs:
                fig.add_trace(go.Scatter3d(
                    x=[spine[0], sat[0]], y=[spine[1], sat[1]], z=[spine[2], sat[2]],
                    mode="lines",
                    line=dict(color=f"rgba({r},{g},{b},0.3)", width=1.5),
                    showlegend=False, hoverinfo="skip",
                ))

            # ---- Paper dots ----
            px = [d[0] for d in papers_sorted]
            py = [d[1] for d in papers_sorted]
            pz = [d[2] for d in papers_sorted]
            p_sizes = [max(4, min(22, np.log1p(d[4]) * 3)) for d in papers_sorted]
            p_hover = [
                f"<b>{d[3]}</b><br>{d[5]} ({d[2]})<br>"
                f"{d[4]:,} citations"
                for d in papers_sorted
            ]

            show_legend = clabel not in legend_seen
            legend_seen.add(clabel)

            fig.add_trace(go.Scatter3d(
                x=px, y=py, z=pz,
                mode="markers",
                name=clabel[:35],
                marker=dict(
                    size=p_sizes, color=color, opacity=0.9,
                    line=dict(width=0.8, color="white"),
                ),
                text=p_hover,
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                showlegend=show_legend,
                legendgroup=clabel,
            ))

            # Shadow projection on z_min plane
            fig.add_trace(go.Scatter3d(
                x=px, y=py, z=[z_min] * len(px),
                mode="markers",
                marker=dict(size=[max(2, s * 0.4) for s in p_sizes],
                            color=color, opacity=0.2),
                showlegend=False, hoverinfo="skip",
            ))

    # Preset view buttons
    fig.update_layout(
        title=f"Research Trajectory: {focal_name}",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(title="Year", range=z_range, dtick=5),
            bgcolor="#1a1a2e",
            dragmode="turntable",
        ),
        paper_bgcolor="#16213e",
        font=dict(color="white"),
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            title="Research Domains",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                x=0.0, y=1.0,
                xanchor="left", yanchor="top",
                buttons=[
                    dict(label="3D View", method="relayout", args=[{
                        "scene.camera": dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                    }]),
                    dict(label="Top Down", method="relayout", args=[{
                        "scene.camera": dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0)),
                    }]),
                    dict(label="Side View", method="relayout", args=[{
                        "scene.camera": dict(eye=dict(x=2.0, y=0, z=0.3)),
                    }]),
                ],
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.1)",
            ),
        ],
    )

    return fig


def render_analysis_markdown(text: str) -> html.Div:
    lines = text.split("\n")
    elements = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            elements.append(html.H6(line[3:], className="mt-3 mb-1 text-primary"))
        elif line.startswith("- **"):
            parts = line[2:].split("**")
            if len(parts) >= 3:
                elements.append(html.P([html.Strong(parts[1]), parts[2]], className="small mb-0 ms-2"))
            else:
                elements.append(html.P(line[2:], className="small mb-0 ms-2"))
        elif line.startswith("- "):
            elements.append(html.P(f"  {line}", className="small mb-0 ms-2"))
        else:
            elements.append(html.P(line, className="small mb-1"))
    return html.Div(elements)


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Research Landscape Map v3",
    suppress_callback_exceptions=True,
)

datasets = discover_datasets()
dataset_options = [{"label": k, "value": str(v)} for k, v in datasets.items()]


def make_layout():
    return dbc.Container([
        # ===== Header =====
        dbc.Row([
            dbc.Col(html.H3("Research Landscape Map", className="my-2"), width=5),
            dbc.Col([
                dbc.InputGroup([
                    dbc.InputGroupText("API Key"),
                    dbc.Input(
                        id="api-key-input",
                        type="password",
                        placeholder="sk-ant-...",
                        persistence=True,
                        persistence_type="local",
                    ),
                    dbc.Button(
                        id="btn-save-key", children="Save",
                        color="outline-secondary", size="sm",
                    ),
                    html.Span(
                        id="api-key-status",
                        className="ms-2 d-flex align-items-center small",
                    ),
                ], size="sm"),
            ], width=4, className="my-2"),
            dbc.Col(dcc.Dropdown(
                id="dataset-dropdown", options=dataset_options,
                value=str(list(datasets.values())[0]) if datasets else None,
                placeholder="Select dataset...", clearable=False,
            ), width=3, className="my-2"),
        ]),

        # ===== AI Intelligent Search =====
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(html.H5("AI Research Assistant", className="mb-0"), width=8),
                    dbc.Col(
                        dbc.Button("Show/Hide", id="btn-toggle-ai-input", color="link", size="sm"),
                        width=4, className="text-end",
                    ),
                ]),
            ),
            dbc.Collapse(
                dbc.CardBody([
                    dbc.Textarea(
                        id="research-idea-input",
                        placeholder=(
                            "Describe your research interest in natural language...\n"
                            "Example: How do online communities on Reddit govern themselves? "
                            "I'm interested in moderation, platform governance, and user participation."
                        ),
                        rows=3, className="mb-2",
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Search Database", id="btn-search",
                                color="primary", className="me-2",
                            ),
                            dbc.Button(
                                "Visualize Results", id="btn-visualize",
                                color="success", className="me-2", disabled=True,
                            ),
                        ], width="auto"),
                        dbc.Col(
                            html.Span(id="search-status", className="text-muted small align-middle"),
                            className="d-flex align-items-center",
                        ),
                    ]),
                    # Search progress + results
                    dcc.Loading(
                        id="search-loading",
                        type="default",
                        children=html.Div(id="search-output", className="mt-3"),
                    ),
                    # Visualization progress
                    dcc.Loading(
                        id="viz-loading",
                        type="default",
                        children=html.Div(id="viz-output", className="mt-2"),
                    ),
                ]),
                id="collapse-ai-input", is_open=True,
            ),
        ], className="mb-3"),

        # ===== Mode Tabs: Paper Landscape / Author Landscape =====
        dbc.Tabs([
            dbc.Tab(label="Paper Landscape", tab_id="mode-papers"),
            dbc.Tab(label="Author Landscape", tab_id="mode-authors"),
        ], id="mode-tabs", active_tab="mode-papers", className="mb-3"),

        # ===== Paper Landscape (shown when mode-papers active) =====
        html.Div(id="paper-landscape-container", children=[
            dbc.Row([
                dbc.Col([
                    # DataMapPlot interactive map (visible)
                    dcc.Loading(
                        type="default",
                        children=html.Iframe(
                            id="datamapplot-iframe",
                            style={"width": "100%", "height": "700px", "border": "none",
                                   "borderRadius": "4px"},
                        ),
                    ),
                    # Hidden Plotly Graph — kept for callback compatibility (fallback)
                    dcc.Graph(
                        id="scatter-plot",
                        config={"scrollZoom": True, "displayModeBar": True,
                                "modeBarButtonsToAdd": ["select2d", "lasso2d"]},
                        style={"display": "none"},
                    ),
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            dbc.Tabs([
                                dbc.Tab(label="Paper Details", tab_id="tab-paper"),
                                dbc.Tab(label="AI Landscape", tab_id="tab-landscape-ai"),
                            ], id="right-panel-tabs", active_tab="tab-paper"),
                        ),
                        dbc.CardBody(id="right-panel-content", style={"maxHeight": "650px", "overflowY": "auto"}),
                    ], className="h-100"),
                ], width=4),
            ]),

            # --- Time Evolution Controls (hidden — datamapplot has semantic zoom) ---
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Time Evolution", className="fw-bold me-3"),
                            dbc.Button("▶ Play", id="btn-play-evolution", color="info",
                                       size="sm", className="me-2"),
                            html.Span(id="evolution-year-label", className="text-info fw-bold ms-2",
                                      style={"fontSize": "1.1em"}),
                        ], width=4, className="d-flex align-items-center"),
                        dbc.Col([
                            dcc.Slider(
                                id="evolution-year-slider",
                                min=1990, max=2026, value=2026,
                                marks={y: str(y) for y in range(1990, 2027, 5)},
                                step=1,
                                tooltip={"placement": "bottom"},
                            ),
                        ], width=8),
                    ]),
                ], className="py-2"),
            ], className="mb-2", color="dark", outline=True,
               style={"display": "none"}),  # hidden — datamapplot replaces time evolution
            dcc.Interval(id="evolution-interval", interval=800, disabled=True),

            dbc.Row([
                dbc.Col([
                    html.Label("Year Range"),
                    dcc.RangeSlider(id="year-slider", min=2000, max=2026, value=[2010, 2026],
                                    marks={y: str(y) for y in range(2000, 2027, 5)},
                                    step=1, tooltip={"placement": "bottom"}),
                ], width=4),
                dbc.Col([
                    html.Label("Min Citations"),
                    dcc.Slider(id="citation-slider", min=0, max=500, value=0, step=10,
                               marks={0: "0", 50: "50", 100: "100", 200: "200", 500: "500"},
                               tooltip={"placement": "bottom"}),
                ], width=3),
                dbc.Col([
                    html.Label("Clusters"),
                    dcc.Dropdown(id="cluster-dropdown", multi=True, placeholder="All clusters"),
                ], width=3),
                dbc.Col([
                    html.Div([
                        dbc.Button("CSV", id="btn-export-csv", color="secondary", size="sm", className="me-1 mt-4"),
                        dbc.Button("BibTeX", id="btn-export-bib", color="secondary", size="sm", className="me-1 mt-4"),
                        dbc.Button("PNG", id="btn-export-png", color="secondary", size="sm", className="mt-4"),
                    ]),
                ], width=2),
            ], className="my-3"),
            dbc.Row([dbc.Col(html.Div(id="stats-bar", className="text-muted small"))]),
        ]),

        # ===== Author Landscape (shown when mode-authors active) =====
        html.Div(id="author-landscape-container", style={"display": "none"}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.Input(
                            id="author-search-input",
                            placeholder="Search author name...",
                            type="text",
                        ),
                        dbc.Button("Search", id="btn-author-search", color="primary"),
                        dbc.Button("Build Author Landscape", id="btn-build-author-landscape",
                                   color="success", className="ms-2", style={"display": "none"}),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Try: ", className="text-muted small me-1"),
                        dbc.Button("Erik Brynjolfsson", id="demo-author-1", color="link", size="sm", className="p-0 me-2"),
                        dbc.Button("Ritu Agarwal", id="demo-author-2", color="link", size="sm", className="p-0 me-2"),
                        dbc.Button("Hsinchun Chen", id="demo-author-3", color="link", size="sm", className="p-0 me-2"),
                        dbc.Button("Paul A. Pavlou", id="demo-author-4", color="link", size="sm", className="p-0 me-2"),
                        dbc.Button("Anindya Ghose", id="demo-author-5", color="link", size="sm", className="p-0"),
                    ], className="mb-2"),
                ], width=8),
                dbc.Col([
                    html.Span(id="author-status", className="text-muted small"),
                ], width=4, className="d-flex align-items-center"),
            ]),

            # --- Multi-Author Compare ---
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Compare Authors (max 5)", className="fw-bold small mb-1"),
                            dcc.Dropdown(
                                id="compare-authors-dropdown",
                                multi=True,
                                placeholder="Search and select authors to compare...",
                                options=[],
                                maxHeight=300,
                            ),
                        ], width=7),
                        dbc.Col([
                            dbc.Button("Add Current Author", id="btn-add-to-compare",
                                       color="outline-info", size="sm", className="me-2 mt-4"),
                            dbc.Button("Compare", id="btn-compare-authors",
                                       color="warning", size="sm", className="mt-4"),
                        ], width=3, className="d-flex align-items-start"),
                        dbc.Col([
                            dbc.Switch(
                                id="toggle-split-view",
                                label="Split View",
                                value=False,
                                className="mt-4",
                            ),
                        ], width=2),
                    ]),
                ], className="py-2"),
            ], className="mb-2", color="dark", outline=True),
            dcc.Loading(
                id="compare-loading",
                type="default",
                children=html.Div(id="compare-output"),
            ),
            dcc.Store(id="compare-authors-store", data=[]),
            dcc.Loading(
                id="author-loading",
                type="default",
                children=html.Div(id="author-build-output"),
            ),
            # Progress bar for ego network building
            html.Div(id="ego-progress-container", style={"display": "none"}, children=[
                html.Div([
                    html.Span(id="ego-progress-label", className="small text-muted me-2"),
                    dbc.Progress(id="ego-progress-bar", value=0, striped=True, animated=True,
                                 className="mb-2", style={"height": "20px"}),
                ]),
            ]),
            dcc.Interval(id="ego-progress-interval", interval=500, disabled=True),

            dbc.Row([
                dbc.Col(
                    dbc.Tabs([
                        dbc.Tab(label="Co-author Network", tab_id="author-view-network"),
                        dbc.Tab(label="Research Territory", tab_id="author-view-territory"),
                        dbc.Tab(label="3D Trajectory", tab_id="author-view-3d"),
                    ], id="author-view-tabs", active_tab="author-view-network"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Switch(
                        id="toggle-subtopic-labels",
                        label="Sub-topic labels",
                        value=False,
                        className="ms-3 mt-1",
                    ),
                    width="auto", className="d-flex align-items-center",
                ),
            ], className="mb-2"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id="author-scatter-plot",
                        config={"scrollZoom": True, "displayModeBar": True},
                        style={"height": "700px"},
                    ),
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Author Details", className="mb-0")),
                        dbc.CardBody(
                            id="author-detail-panel",
                            style={"maxHeight": "650px", "overflowY": "auto"},
                            children=html.P("Click an author on the map or search by name.",
                                            className="text-muted"),
                        ),
                    ], className="h-100"),
                ], width=4),
            ]),
            dbc.Row([dbc.Col(html.Div(id="author-stats-bar", className="text-muted small"))]),
        ]),

        # Hidden components
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-bib"),
        dcc.Download(id="download-png"),
        dcc.Store(id="current-data-path"),
        dcc.Store(id="ai-landscape-cache", data=""),
        dcc.Store(id="search-results-store", data=None),
        dcc.Store(id="author-landscape-ready", data=False),
        dcc.Store(id="selected-author-id", data=None),
        dcc.Store(id="pending-author-id", data=None),
        html.Div(dbc.Button(id="btn-confirm-author", style={"display": "none"})),
    ], fluid=True, className="py-3")


app.layout = make_layout


# ===========================================================================
# Callbacks
# ===========================================================================

# Toggle AI panel
@app.callback(
    Output("collapse-ai-input", "is_open"),
    Input("btn-toggle-ai-input", "n_clicks"),
    State("collapse-ai-input", "is_open"),
    prevent_initial_call=True,
)
def toggle_ai_input(n, is_open):
    return not is_open


# API Key: status indicator on load + after save
@app.callback(
    Output("api-key-status", "children"),
    Output("api-key-input", "value"),
    Input("btn-save-key", "n_clicks"),
    State("api-key-input", "value"),
)
def api_key_handler(n_clicks, input_value):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    env_path = PROJECT_ROOT / ".env"

    # On save button click: write to .env
    if "btn-save-key" in triggered and input_value and input_value.strip():
        key = input_value.strip()
        # Read existing .env, update or add ANTHROPIC_API_KEY
        lines = []
        found = False
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    lines.append(f"ANTHROPIC_API_KEY={key}")
                    found = True
                else:
                    lines.append(line)
        if not found:
            lines.append(f"ANTHROPIC_API_KEY={key}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.environ["ANTHROPIC_API_KEY"] = key

        return (
            html.Span("Saved", className="text-success"),
            key,
        )

    # On page load: check if key exists in .env or input
    saved_key = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                saved_key = line.split("=", 1)[1].strip()
                break

    if input_value and input_value.strip():
        # User has typed a key (from localStorage persistence)
        return html.Span("Ready", className="text-success"), input_value
    elif saved_key:
        # Key in .env but not in input — fill it in
        return html.Span("Loaded from .env", className="text-success"), saved_key
    else:
        return html.Span("Not set", className="text-warning"), ""


# Store dataset path
@app.callback(Output("current-data-path", "data"), Input("dataset-dropdown", "value"))
def store_dataset(path):
    return path


# ---------------------------------------------------------------------------
# Intelligent Search callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("search-output", "children"),
    Output("search-status", "children"),
    Output("search-results-store", "data"),
    Output("btn-visualize", "disabled"),
    Input("btn-search", "n_clicks"),
    State("research-idea-input", "value"),
    State("api-key-input", "value"),
    prevent_initial_call=True,
)
def run_intelligent_search(n_clicks, idea, api_key):
    if not n_clicks or not idea or not idea.strip():
        raise PreventUpdate

    if not api_key or not api_key.strip():
        return (
            dbc.Alert("Please enter your Anthropic API key above.", color="warning"),
            "Missing API key",
            None,
            True,
        )

    try:
        from analyze import intelligent_search

        result = intelligent_search(
            research_idea=idea,
            api_key=api_key.strip(),
            max_papers=200,
        )

        papers = result["papers"]
        search_log = result["search_log"]
        summary = result["summary"]

        if not papers:
            return (
                dbc.Alert("No papers found. Try different keywords.", color="warning"),
                "No results",
                None,
                True,
            )

        # Build result display
        children = []

        # Summary from Claude
        if summary:
            children.append(dbc.Alert([
                html.Strong(f"Found {len(papers)} unique papers "),
                html.Span(f"from {len(search_log)} searches", className="text-muted"),
            ], color="info", className="py-2"))
            children.append(render_analysis_markdown(summary))

        # Search log
        children.append(html.Hr())
        children.append(html.H6("Search Queries:", className="text-muted"))
        for i, log in enumerate(search_log):
            kw_str = ", ".join(log["keywords"])
            filters_str = ", ".join(f"{k}={v}" for k, v in log["filters"].items() if v)
            children.append(html.P(
                f"{i+1}. [{kw_str}] → {log['results']} hits, {log['new']} new"
                + (f" | {filters_str}" if filters_str else ""),
                className="small mb-0 ms-2 text-muted",
            ))

        # Top papers preview
        children.append(html.Hr())
        children.append(html.H6("Top papers by citation:"))
        top = sorted(papers, key=lambda p: p.get("citation_count", 0), reverse=True)[:10]
        for p in top:
            tier_badge = ""
            if p.get("journal_tier", "").startswith("Tier1"):
                tier_badge = "🏆 "
            elif p.get("journal_tier", "").startswith("Tier2"):
                tier_badge = "⭐ "
            children.append(html.P([
                html.Strong(f"{tier_badge}{p['title'][:80]}"),
                html.Br(),
                html.Span(
                    f"{p.get('journal', '')} ({p['year']}) — {p['citation_count']} citations",
                    className="text-muted",
                ),
            ], className="small mb-1"))

        return (
            html.Div(children),
            f"{len(papers)} papers found",
            papers,  # Store for visualization
            False,  # Enable visualize button
        )

    except ValueError as e:
        return (
            dbc.Alert(str(e), color="danger"),
            "API key error",
            None,
            True,
        )
    except Exception as e:
        traceback.print_exc()
        return (
            dbc.Alert(f"Error: {type(e).__name__}: {e}", color="danger"),
            "Failed",
            None,
            True,
        )


# ---------------------------------------------------------------------------
# Visualize search results: embed + cluster → update scatter
# ---------------------------------------------------------------------------
@app.callback(
    Output("viz-output", "children"),
    Output("dataset-dropdown", "options", allow_duplicate=True),
    Output("dataset-dropdown", "value", allow_duplicate=True),
    Input("btn-visualize", "n_clicks"),
    State("search-results-store", "data"),
    prevent_initial_call=True,
)
def visualize_search_results(n_clicks, papers):
    if not n_clicks or not papers:
        raise PreventUpdate

    try:
        t0 = time.time()
        print(f"[VISUALIZE] Embedding + clustering {len(papers)} papers...")

        df = embed_and_cluster_papers(papers)

        # Save as clustered CSV
        out_path = DATA_DIR / "search_clustered.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        elapsed = int(time.time() - t0)
        print(f"[VISUALIZE] Done in {elapsed}s: {len(df)} papers, {df['cluster_id'].nunique()} clusters")

        # Refresh dataset list
        ds = discover_datasets()
        ds_options = [{"label": k, "value": str(v)} for k, v in ds.items()]

        return (
            dbc.Alert(
                f"Visualization ready! {len(df)} papers, "
                f"{df['cluster_id'].nunique()} clusters ({elapsed}s)",
                color="success", className="py-2",
            ),
            ds_options,
            str(out_path),
        )
    except Exception as e:
        traceback.print_exc()
        return (
            dbc.Alert(f"Visualization failed: {type(e).__name__}: {e}", color="danger"),
            no_update,
            no_update,
        )


# ---------------------------------------------------------------------------
# Update scatter plot
# ---------------------------------------------------------------------------
@app.callback(
    Output("scatter-plot", "figure"),
    Output("datamapplot-iframe", "src"),
    Output("cluster-dropdown", "options"),
    Output("stats-bar", "children"),
    Output("year-slider", "min"),
    Output("year-slider", "max"),
    Output("year-slider", "value"),
    Input("current-data-path", "data"),
    Input("year-slider", "value"),
    Input("citation-slider", "value"),
    Input("cluster-dropdown", "value"),
)
def update_scatter(data_path, year_range, min_citations, selected_clusters):
    if not data_path:
        raise PreventUpdate

    df = load_dataset(Path(data_path))
    user_papers = load_user_papers()

    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    is_dataset_change = "current-data-path" in triggered or not triggered

    year_min_data = int(df["year"].min()) if not df.empty else 2000
    year_max_data = int(df["year"].max()) if not df.empty else 2026
    yr = [year_min_data, year_max_data] if is_dataset_change else (year_range or [year_min_data, year_max_data])

    mask = (df["year"] >= yr[0]) & (df["year"] <= yr[1]) & (df["citation_count"] >= (min_citations or 0))
    if selected_clusters:
        mask &= df["cluster_label"].isin(selected_clusters)
    filtered = df[mask]

    cluster_options = [
        {"label": f"{c} ({(df['cluster_label'] == c).sum()})", "value": c}
        for c in sorted(df["cluster_label"].unique())
    ]

    # Hidden Plotly figure (for time-evolution callbacks compatibility)
    fig = build_territory_scatter(df, filtered, user_papers, data_path=data_path)

    # DataMapPlot interactive HTML — the visible visualization
    cache_key = f"{data_path}_{yr}_{min_citations}_{selected_clusters}"
    try:
        iframe_src = _generate_datamapplot_html(filtered, cache_key=cache_key)
    except Exception as e:
        traceback.print_exc()
        iframe_src = no_update  # keep old map if generation fails

    n_fields = _assign_territories(df, level="field").nunique()
    stats = f"{len(filtered):,} papers | {n_fields} territories | {yr[0]}-{yr[1]}"

    return fig, iframe_src, cluster_options, stats, year_min_data, year_max_data, yr


# ---------------------------------------------------------------------------
# Right panel (tabs)
# ---------------------------------------------------------------------------
@app.callback(
    Output("right-panel-content", "children"),
    Input("right-panel-tabs", "active_tab"),
    Input("scatter-plot", "clickData"),
    State("current-data-path", "data"),
    State("ai-landscape-cache", "data"),
    State("api-key-input", "value"),
)
def update_right_panel(active_tab, click_data, data_path, landscape_cache, api_key):
    if not data_path:
        raise PreventUpdate

    if active_tab == "tab-paper":
        return _render_paper_details(click_data, data_path)
    elif active_tab == "tab-landscape-ai":
        return _render_landscape_ai_tab(data_path, landscape_cache)
    return html.P("Select a tab.", className="text-muted")


def _render_paper_details(click_data, data_path):
    if not click_data:
        return html.P("Click a paper on the map to see details.", className="text-muted")

    point = click_data["points"][0]
    idx = point.get("customdata")
    if idx is None:
        return html.P("No data.", className="text-muted")

    df = load_dataset(Path(data_path))
    if idx >= len(df):
        return html.P("Paper not found.", className="text-muted")

    row = df.iloc[idx]
    doi = str(row.get("doi", ""))
    doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}" if doi else ""

    details = [
        html.H6(row.get("title", "Untitled"), className="mb-2"),
        html.P([html.Strong("Authors: "), str(row.get("authors", "N/A"))[:200]], className="small mb-1"),
        html.P([
            html.Strong("Year: "), str(row.get("year", "")),
            html.Span(" | ", className="mx-1"),
            html.Strong("Journal: "), str(row.get("journal", "")),
        ], className="small mb-1"),
        html.P([
            html.Strong("Citations: "),
            html.Span(f"{row.get('citation_count', 0):,}", className="text-primary fw-bold"),
        ], className="small mb-1"),
        html.P([
            html.Strong("Cluster: "),
            html.Span(str(row.get("cluster_label", "")), className="badge bg-info text-dark"),
        ], className="small mb-1"),
    ]

    # Show journal tier if available
    tier = str(row.get("journal_tier", ""))
    if tier:
        details.append(html.P([html.Strong("Tier: "), tier], className="small mb-1"))

    details.extend([
        html.Hr(),
        html.P(
            str(row.get("abstract", ""))[:500] + ("..." if len(str(row.get("abstract", ""))) > 500 else ""),
            className="small", style={"maxHeight": "200px", "overflowY": "auto"},
        ),
    ])
    if doi_url:
        details.append(html.A(
            dbc.Button("Open DOI", color="primary", size="sm", className="mt-2"),
            href=doi_url, target="_blank",
        ))
    return details


def _render_landscape_ai_tab(data_path, landscape_cache):
    if landscape_cache:
        return html.Div([
            html.H6("Landscape Overview", className="text-primary"),
            html.Hr(),
            render_analysis_markdown(landscape_cache),
        ])

    return html.Div([
        html.P("Generate an AI overview of the entire research landscape.", className="text-muted"),
        html.Ul([
            html.Li("How clusters relate to each other", className="small"),
            html.Li("Field evolution & cross-cluster connections", className="small"),
            html.Li("Underexplored areas & gaps", className="small"),
        ]),
        dbc.Button("Analyze full landscape with AI", id="btn-analyze-landscape", color="success", size="sm"),
        html.Div(id="landscape-analysis-output", className="mt-3"),
    ])


# ---------------------------------------------------------------------------
# AI Landscape Analysis callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("landscape-analysis-output", "children"),
    Output("ai-landscape-cache", "data"),
    Input("btn-analyze-landscape", "n_clicks"),
    State("current-data-path", "data"),
    State("ai-landscape-cache", "data"),
    State("api-key-input", "value"),
    prevent_initial_call=True,
)
def run_landscape_analysis(n_clicks, data_path, cache, api_key):
    if not n_clicks:
        raise PreventUpdate
    try:
        from analyze import analyze_landscape
        df = load_dataset(Path(data_path))
        analysis = analyze_landscape(df, api_key=api_key)
        return render_analysis_markdown(analysis), analysis
    except Exception as e:
        return html.P(f"Error: {e}", className="text-danger small"), ""


# ---------------------------------------------------------------------------
# Export callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("download-csv", "data"), Input("btn-export-csv", "n_clicks"),
    State("current-data-path", "data"), State("year-slider", "value"),
    State("citation-slider", "value"), State("cluster-dropdown", "value"),
    prevent_initial_call=True,
)
def export_csv(n, path, yr, cit, cl):
    if not n or not path: raise PreventUpdate
    return dcc.send_data_frame(_get_filtered_df(path, yr, cit, cl).to_csv, "landscape_export.csv", index=False)


@app.callback(
    Output("download-bib", "data"), Input("btn-export-bib", "n_clicks"),
    State("current-data-path", "data"), State("year-slider", "value"),
    State("citation-slider", "value"), State("cluster-dropdown", "value"),
    prevent_initial_call=True,
)
def export_bibtex(n, path, yr, cit, cl):
    if not n or not path: raise PreventUpdate
    return dict(content=to_bibtex(_get_filtered_df(path, yr, cit, cl)), filename="landscape_export.bib")


@app.callback(
    Output("download-png", "data"), Input("btn-export-png", "n_clicks"),
    State("scatter-plot", "figure"), prevent_initial_call=True,
)
def export_png(n, figure):
    if not n or not figure: raise PreventUpdate
    try:
        fig = go.Figure(figure)
        fig.update_layout(width=1600, height=1200)
        img_bytes = fig.to_image(format="png", scale=2, engine="kaleido")
        (OUTPUT_DIR / "landscape_export.png").write_bytes(img_bytes)
        return dcc.send_bytes(img_bytes, "landscape_300dpi.png")
    except Exception:
        return no_update


# ---------------------------------------------------------------------------
# Time Evolution: Play/Pause + Interval + Bubble overlay
# ---------------------------------------------------------------------------
@app.callback(
    Output("evolution-interval", "disabled"),
    Output("btn-play-evolution", "children"),
    Output("evolution-year-slider", "value", allow_duplicate=True),
    Input("btn-play-evolution", "n_clicks"),
    State("evolution-interval", "disabled"),
    State("evolution-year-slider", "min"),
    State("evolution-year-slider", "value"),
    State("evolution-year-slider", "max"),
    prevent_initial_call=True,
)
def toggle_play_evolution(n_clicks, is_disabled, yr_min, yr_val, yr_max):
    if not n_clicks:
        raise PreventUpdate
    if is_disabled:
        # Start playing: reset to min year if at max
        start_yr = yr_min if yr_val >= yr_max else yr_val
        return False, "⏸ Pause", start_yr
    else:
        return True, "▶ Play", yr_val


@app.callback(
    Output("evolution-year-slider", "value", allow_duplicate=True),
    Output("evolution-interval", "disabled", allow_duplicate=True),
    Output("btn-play-evolution", "children", allow_duplicate=True),
    Input("evolution-interval", "n_intervals"),
    State("evolution-year-slider", "value"),
    State("evolution-year-slider", "max"),
    prevent_initial_call=True,
)
def advance_evolution_year(n_intervals, current_year, max_year):
    if current_year is None:
        raise PreventUpdate
    next_year = current_year + 1
    if next_year > max_year:
        return max_year, True, "▶ Play"
    return next_year, False, "⏸ Pause"


@app.callback(
    Output("scatter-plot", "figure", allow_duplicate=True),
    Output("evolution-year-label", "children"),
    Output("evolution-year-slider", "min"),
    Output("evolution-year-slider", "max", allow_duplicate=True),
    Input("evolution-year-slider", "value"),
    State("current-data-path", "data"),
    prevent_initial_call=True,
)
def update_evolution_bubbles(evo_year, data_path):
    if not data_path or evo_year is None:
        raise PreventUpdate

    df = load_dataset(Path(data_path))
    if df.empty:
        raise PreventUpdate

    yr_min = int(df["year"].min())
    yr_max = int(df["year"].max())
    filtered = df[df["year"] <= evo_year]

    fig = build_territory_scatter(
        df, filtered, data_path=data_path,
        title=f"Research Landscape — up to {evo_year}",
    )

    label = f"Showing up to {evo_year} ({len(filtered):,} papers)"
    return fig, label, yr_min, yr_max


# ---------------------------------------------------------------------------
# Mode tab: show/hide Paper vs Author landscape + auto-load cache
# ---------------------------------------------------------------------------
@app.callback(
    Output("paper-landscape-container", "style"),
    Output("author-landscape-container", "style"),
    Output("author-scatter-plot", "figure", allow_duplicate=True),
    Output("author-stats-bar", "children", allow_duplicate=True),
    Output("author-landscape-ready", "data", allow_duplicate=True),
    Input("mode-tabs", "active_tab"),
    State("author-landscape-ready", "data"),
    prevent_initial_call=True,
)
def switch_mode(active_tab, already_ready):
    if active_tab == "mode-authors":
        # Auto-load from cache if available and not already loaded
        if not already_ready:
            csv_path = DATA_DIR / "authors_clustered.csv"
            if csv_path.exists():
                author_df = pd.read_csv(csv_path, encoding="utf-8-sig")
                fig = build_author_scatter(author_df)
                stats = f"{len(author_df):,} authors | {author_df['cluster_id'].nunique()} groups | Loaded from cache"
                return {"display": "none"}, {"display": "block"}, fig, stats, True
        return {"display": "none"}, {"display": "block"}, no_update, no_update, no_update
    return {"display": "block"}, {"display": "none"}, no_update, no_update, no_update


# ---------------------------------------------------------------------------
# Build Author Landscape: embed + cluster authors
# ---------------------------------------------------------------------------
@app.callback(
    Output("author-build-output", "children"),
    Output("author-scatter-plot", "figure"),
    Output("author-stats-bar", "children"),
    Output("author-landscape-ready", "data"),
    Input("btn-build-author-landscape", "n_clicks"),
    prevent_initial_call=True,
)
def build_author_landscape(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    try:
        # Check if author tables exist
        db_path = DATA_DIR / "openalex.db"
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()

        if "authors" not in tables or "paper_authors" not in tables:
            return (
                dbc.Alert(
                    "Author tables not built yet. Run: python build_db.py --build-authors",
                    color="warning",
                ),
                go.Figure(),
                "",
                False,
            )

        t0 = time.time()
        print("[AUTHOR] Building author landscape...")

        from embed import generate_author_embeddings
        from cluster import cluster_authors

        # Generate author embeddings (cached)
        author_embeddings, author_ids, author_df = generate_author_embeddings(min_papers=3)

        if len(author_df) == 0:
            return (
                dbc.Alert("No authors with >= 3 papers found.", color="warning"),
                go.Figure(),
                "",
                False,
            )

        # Cluster authors
        author_df = cluster_authors(author_df, author_embeddings)

        # Save for reuse
        author_df.to_csv(DATA_DIR / "authors_clustered.csv", index=False, encoding="utf-8-sig")

        elapsed = int(time.time() - t0)
        n_authors = len(author_df)
        n_clusters = author_df["cluster_id"].nunique()

        fig = build_author_scatter(author_df)
        stats = f"{n_authors:,} authors | {n_clusters} groups | Built in {elapsed}s"

        print(f"[AUTHOR] Done: {n_authors} authors, {n_clusters} clusters, {elapsed}s")

        return (
            dbc.Alert(f"Author landscape ready! {n_authors:,} authors, {n_clusters} groups.",
                      color="success", className="py-2"),
            fig,
            stats,
            True,
        )

    except Exception as e:
        traceback.print_exc()
        return (
            dbc.Alert(f"Error: {type(e).__name__}: {e}", color="danger"),
            go.Figure(),
            "",
            False,
        )


# ---------------------------------------------------------------------------
# Demo author buttons: directly select known author_id (skip search)
# ---------------------------------------------------------------------------
DEMO_AUTHORS = {
    "demo-author-1": ("Erik Brynjolfsson", "https://openalex.org/A5038255653"),
    "demo-author-2": ("Ritu Agarwal", "https://openalex.org/A5081578599"),
    "demo-author-3": ("Hsinchun Chen", "https://openalex.org/A5109924510"),
    "demo-author-4": ("Paul A. Pavlou", "https://openalex.org/A5008496192"),
    "demo-author-5": ("Anindya Ghose", "https://openalex.org/A5073770532"),
}


@app.callback(
    Output("selected-author-id", "data"),
    Output("author-search-input", "value"),
    Input("demo-author-1", "n_clicks"),
    Input("demo-author-2", "n_clicks"),
    Input("demo-author-3", "n_clicks"),
    Input("demo-author-4", "n_clicks"),
    Input("demo-author-5", "n_clicks"),
    prevent_initial_call=True,
)
def demo_author_click(n1, n2, n3, n4, n5):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    btn_id = triggered.split(".")[0]
    if btn_id not in DEMO_AUTHORS:
        raise PreventUpdate
    name, aid = DEMO_AUTHORS[btn_id]
    return aid, name


# ---------------------------------------------------------------------------
# Author search step 1: show candidate list for disambiguation
# ---------------------------------------------------------------------------
@app.callback(
    Output("author-detail-panel", "children"),
    Output("author-status", "children"),
    Input("btn-author-search", "n_clicks"),
    State("author-search-input", "value"),
    prevent_initial_call=True,
)
def author_search(n_clicks, search_name):
    if not n_clicks or not search_name or not search_name.strip():
        raise PreventUpdate

    query = search_name.strip()
    db_path = DATA_DIR / "openalex.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Find candidates ranked by citations
    candidates = conn.execute(
        "SELECT author_id, name, institution, paper_count, total_citations "
        "FROM authors WHERE name LIKE ? AND paper_count >= 2 "
        "ORDER BY total_citations DESC LIMIT 10",
        (f"%{query}%",)
    ).fetchall()

    if not candidates:
        conn.close()
        return (
            html.P(f"No author found matching '{query}'.", className="text-warning"),
            f"No match for '{query}'",
        )

    # For each candidate, get their top-cited paper
    items = []
    items.append(html.H6(f"Select the correct author ({len(candidates)} matches):", className="mb-3"))

    for c in candidates:
        top_paper = conn.execute(
            "SELECT p.title, p.year, p.citation_count, p.journal "
            "FROM paper_authors pa JOIN papers p ON p.openalex_id = pa.paper_id "
            "WHERE pa.author_id = ? ORDER BY p.citation_count DESC LIMIT 1",
            (c["author_id"],)
        ).fetchone()

        top_paper_text = ""
        if top_paper:
            top_paper_text = (
                f"{top_paper['title'][:65]}... "
                f"({top_paper['journal']}, {top_paper['year']}, "
                f"{top_paper['citation_count']:,} cites)"
            )

        items.append(
            html.Div(
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Strong(c["name"], className="me-2"),
                            dbc.Badge(f"{c['total_citations']:,} cites", color="primary", className="me-1"),
                            dbc.Badge(f"{c['paper_count']} papers", color="secondary"),
                        ]),
                        html.P(c["institution"] or "Institution unknown",
                               className="small text-muted mb-1"),
                        html.P(
                            top_paper_text,
                            className="small text-info mb-0",
                            style={"fontStyle": "italic"},
                        ) if top_paper_text else None,
                    ], className="py-2 px-3"),
                ],
                    className="mb-2",
                    style={"cursor": "pointer"},
                    color="dark",
                ),
                id={"type": "author-candidate", "id": c["author_id"]},
            )
        )

    conn.close()
    return html.Div(items), f"{len(candidates)} matches for '{query}'"


# ---------------------------------------------------------------------------
# Author search step 2: user selects a candidate → show details + highlight
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Author view toggle: network vs territory
# ---------------------------------------------------------------------------
@app.callback(
    Output("author-scatter-plot", "figure", allow_duplicate=True),
    Input("author-view-tabs", "active_tab"),
    prevent_initial_call=True,
)
def toggle_author_view(active_tab):
    empty = go.Figure(layout=dict(
        title="Select an author first",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="white"),
    ))
    if active_tab == "author-view-territory":
        return _ego_progress.get("result_territory") or empty
    elif active_tab == "author-view-3d":
        return _ego_progress.get("result_3d") or empty
    else:
        result_fig = _ego_progress.get("result_fig")
        if result_fig:
            return result_fig
        return no_update


# ---------------------------------------------------------------------------
# Toggle sub-topic labels visibility on territory plot
# ---------------------------------------------------------------------------
@app.callback(
    Output("author-scatter-plot", "figure", allow_duplicate=True),
    Input("toggle-subtopic-labels", "value"),
    State("author-view-tabs", "active_tab"),
    prevent_initial_call=True,
)
def toggle_subtopic_labels(show_subtopics, active_tab):
    if active_tab != "author-view-territory":
        raise PreventUpdate

    territory_fig = _ego_progress.get("result_territory")
    if not territory_fig:
        raise PreventUpdate

    # Update annotation visibility: fine-level labels have font.size=9
    fig = go.Figure(territory_fig)
    if fig.layout.annotations:
        new_annotations = []
        for ann in fig.layout.annotations:
            ann_dict = ann.to_plotly_json()
            # Fine-level labels have size 9, coarse have size 13
            if ann_dict.get("font", {}).get("size") == 9:
                ann_dict["visible"] = show_subtopics
            new_annotations.append(ann_dict)
        fig.update_layout(annotations=new_annotations)

    # Store the updated figure back
    _ego_progress["result_territory"] = fig
    return fig


# ---------------------------------------------------------------------------
# Multi-Author Compare: Add current author to compare list
# ---------------------------------------------------------------------------
@app.callback(
    Output("compare-authors-dropdown", "options"),
    Output("compare-authors-dropdown", "value"),
    Output("compare-authors-store", "data"),
    Input("btn-add-to-compare", "n_clicks"),
    State("selected-author-id", "data"),
    State("compare-authors-store", "data"),
    State("compare-authors-dropdown", "value"),
    prevent_initial_call=True,
)
def add_author_to_compare(n_clicks, author_id, compare_store, current_selection):
    if not n_clicks or not author_id:
        raise PreventUpdate

    compare_store = compare_store or []
    current_selection = current_selection or []

    # Check limit
    if len(current_selection) >= 5:
        raise PreventUpdate

    # Check if already added
    existing_ids = [a["id"] for a in compare_store]
    if author_id in existing_ids:
        raise PreventUpdate

    # Look up author name
    db_path = DATA_DIR / "openalex.db"
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT name, institution, total_citations FROM authors WHERE author_id = ?",
        (author_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise PreventUpdate

    name = row[0]
    new_entry = {"id": author_id, "name": name, "institution": row[1] or "", "citations": row[2]}
    compare_store.append(new_entry)

    options = [{"label": f"{a['name']} ({a['citations']:,} cites)", "value": a["id"]}
               for a in compare_store]
    new_selection = current_selection + [author_id]

    return options, new_selection, compare_store


# ---------------------------------------------------------------------------
# Multi-Author Compare: Build comparison visualization
# ---------------------------------------------------------------------------
AUTHOR_SYMBOLS = ["circle", "diamond", "square", "triangle-up", "cross"]
AUTHOR_BORDERS = ["gold", "#FF6692", "#19D3F3", "#B6E880", "#FF97FF"]


@app.callback(
    Output("compare-output", "children"),
    Output("author-scatter-plot", "figure", allow_duplicate=True),
    Output("author-status", "children", allow_duplicate=True),
    Input("btn-compare-authors", "n_clicks"),
    State("compare-authors-dropdown", "value"),
    State("compare-authors-store", "data"),
    State("toggle-split-view", "value"),
    prevent_initial_call=True,
)
def compare_authors(n_clicks, selected_ids, compare_store, split_view):
    if not n_clicks or not selected_ids or len(selected_ids) < 2:
        if selected_ids and len(selected_ids) < 2:
            return (
                dbc.Alert("Select at least 2 authors to compare.", color="warning", className="py-2"),
                no_update, no_update,
            )
        raise PreventUpdate

    if len(selected_ids) > 5:
        return (
            dbc.Alert("Maximum 5 authors allowed.", color="warning", className="py-2"),
            no_update, no_update,
        )

    try:
        t0 = time.time()
        author_map = {a["id"]: a for a in (compare_store or []) if a["id"] in selected_ids}

        # Collect papers for each author
        db_path = DATA_DIR / "openalex.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        author_papers = {}  # author_id -> [paper dicts]
        all_paper_ids = set()
        paper_to_authors = {}  # paper_id -> set of author_ids

        for aid in selected_ids:
            papers = conn.execute("""
                SELECT p.openalex_id, p.title, p.abstract, p.year, p.citation_count,
                       p.journal, p.journal_tier, p.topic, p.subfield
                FROM paper_authors pa JOIN papers p ON p.openalex_id = pa.paper_id
                WHERE pa.author_id = ?
            """, (aid,)).fetchall()
            author_papers[aid] = [dict(p) for p in papers]
            for p in papers:
                pid = p["openalex_id"]
                all_paper_ids.add(pid)
                if pid not in paper_to_authors:
                    paper_to_authors[pid] = set()
                paper_to_authors[pid].add(aid)

        conn.close()

        # Find co-authored papers (shared by 2+ selected authors)
        shared_paper_ids = {pid for pid, authors in paper_to_authors.items() if len(authors) >= 2}

        # Deduplicate papers for embedding
        seen = set()
        unique_papers = []
        paper_owner = []  # which author_id "owns" each unique paper (first seen)
        for aid in selected_ids:
            for p in author_papers[aid]:
                pid = p["openalex_id"]
                if pid not in seen:
                    seen.add(pid)
                    unique_papers.append(p)
                    paper_owner.append(aid)

        if len(unique_papers) < 5:
            return (
                dbc.Alert("Not enough papers to compare (need >= 5).", color="warning"),
                no_update, no_update,
            )

        # Embed all papers in one shared UMAP space
        from embed import generate_embeddings_from_texts
        from cluster import cluster_from_embeddings

        texts = [f"{p['title']}. {(p['abstract'] or '')[:300]}" for p in unique_papers]
        embeddings = generate_embeddings_from_texts(texts)

        # Build DataFrame for clustering
        df = pd.DataFrame(unique_papers)
        df = cluster_from_embeddings(df, embeddings)

        # Map paper_id to row index
        pid_to_idx = {p["openalex_id"]: i for i, p in enumerate(unique_papers)}

        palette = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        ]
        cluster_labels = sorted(df["cluster_label"].unique())
        cluster_colors = {c: palette[i % len(palette)] for i, c in enumerate(cluster_labels)}

        if split_view and len(selected_ids) >= 2:
            # --- Split View: one subplot per author ---
            from plotly.subplots import make_subplots
            n_authors = len(selected_ids)
            fig = make_subplots(
                rows=1, cols=n_authors,
                shared_xaxes=True, shared_yaxes=True,
                subplot_titles=[author_map.get(aid, {}).get("name", aid)[:25] for aid in selected_ids],
                horizontal_spacing=0.02,
            )

            for col_idx, aid in enumerate(selected_ids, 1):
                # This author's papers
                a_pids = {p["openalex_id"] for p in author_papers[aid]}
                a_indices = [pid_to_idx[pid] for pid in a_pids if pid in pid_to_idx]
                a_df = df.iloc[a_indices] if a_indices else pd.DataFrame()

                if not a_df.empty:
                    sizes = np.log1p(a_df["citation_count"].values)
                    sizes = (sizes / max(sizes.max(), 1)) * 20 + 4
                    c_colors_list = [cluster_colors.get(cl, "#636EFA") for cl in a_df["cluster_label"]]

                    fig.add_trace(go.Scatter(
                        x=a_df["x"], y=a_df["y"], mode="markers",
                        marker=dict(
                            size=sizes,
                            color=c_colors_list,
                            opacity=0.7,
                            symbol=AUTHOR_SYMBOLS[col_idx - 1],
                            line=dict(width=2, color=AUTHOR_BORDERS[col_idx - 1]),
                        ),
                        text=a_df["title"],
                        hovertemplate="<b>%{text}</b><extra></extra>",
                        showlegend=False,
                    ), row=1, col=col_idx)

                    # Co-authored papers as stars in each panel
                    coauth_idx = [pid_to_idx[pid] for pid in shared_paper_ids
                                  if pid in pid_to_idx and pid in a_pids]
                    if coauth_idx:
                        ca_df = df.iloc[coauth_idx]
                        fig.add_trace(go.Scatter(
                            x=ca_df["x"], y=ca_df["y"], mode="markers",
                            marker=dict(size=18, symbol="star", color="white",
                                        line=dict(width=2, color="gold")),
                            text=ca_df["title"],
                            hovertemplate="<b>%{text}</b><extra>Co-authored</extra>",
                            showlegend=False,
                        ), row=1, col=col_idx)

            for i in range(1, n_authors + 1):
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=i)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=i)

            fig.update_layout(
                title="Multi-Author Comparison (Split View)",
                plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", height=700,
                font=dict(color="white"),
                margin=dict(l=20, r=20, t=60, b=20),
            )

        else:
            # --- Overlay View: all authors on same plot ---
            fig = go.Figure()

            for i, aid in enumerate(selected_ids):
                a_name = author_map.get(aid, {}).get("name", f"Author {i+1}")
                a_pids = {p["openalex_id"] for p in author_papers[aid]}
                # Exclude shared papers from individual traces (will add them separately)
                solo_pids = a_pids - shared_paper_ids
                a_indices = [pid_to_idx[pid] for pid in solo_pids if pid in pid_to_idx]
                a_df = df.iloc[a_indices] if a_indices else pd.DataFrame()

                if not a_df.empty:
                    sizes = np.log1p(a_df["citation_count"].values)
                    sizes = (sizes / max(sizes.max(), 1)) * 20 + 4
                    c_colors_list = [cluster_colors.get(cl, "#636EFA") for cl in a_df["cluster_label"]]

                    fig.add_trace(go.Scatter(
                        x=a_df["x"], y=a_df["y"], mode="markers",
                        name=a_name,
                        legendgroup=a_name,
                        marker=dict(
                            size=sizes,
                            color=c_colors_list,
                            opacity=0.8,
                            symbol=AUTHOR_SYMBOLS[i],
                            line=dict(width=2, color=AUTHOR_BORDERS[i]),
                        ),
                        text=a_df["title"],
                        hovertemplate="<b>%{text}</b><extra>" + a_name + "</extra>",
                    ))

            # Co-authored papers as white stars
            if shared_paper_ids:
                coauth_idx = [pid_to_idx[pid] for pid in shared_paper_ids if pid in pid_to_idx]
                if coauth_idx:
                    ca_df = df.iloc[coauth_idx]
                    # Build hover showing which authors share the paper
                    coauth_hover = []
                    for _, row in ca_df.iterrows():
                        pid = row["openalex_id"]
                        author_names = [author_map.get(a, {}).get("name", "?")
                                       for a in paper_to_authors.get(pid, set())
                                       if a in selected_ids]
                        coauth_hover.append(
                            f"<b>{row['title'][:60]}</b><br>"
                            f"Shared by: {', '.join(author_names)}"
                        )
                    fig.add_trace(go.Scatter(
                        x=ca_df["x"], y=ca_df["y"], mode="markers",
                        name="Co-authored",
                        marker=dict(size=18, symbol="star", color="white",
                                    line=dict(width=2, color="gold")),
                        text=coauth_hover,
                        hovertemplate="%{text}<extra>Co-authored</extra>",
                    ))

            # Add cluster labels at cluster centroids
            annotations = []
            for cl in df["cluster_label"].unique():
                cl_mask = df["cluster_label"] == cl
                if cl_mask.sum() < 3:
                    continue
                cx = df.loc[cl_mask, "x"].mean()
                cy = df.loc[cl_mask, "y"].mean()
                annotations.append(dict(
                    x=cx, y=cy, text=cl[:30], showarrow=False,
                    font=dict(size=9, color="rgba(255,255,255,0.6)"),
                    xanchor="center", yanchor="middle",
                ))

            names = [author_map.get(aid, {}).get("name", "?") for aid in selected_ids]
            fig.update_layout(
                title=f"Author Comparison: {' vs '.join(n[:20] for n in names)} ({len(unique_papers)} papers)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", height=700,
                font=dict(color="white"),
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(title="Authors", font=dict(size=10), itemsizing="constant"),
                clickmode="event",
                annotations=annotations,
            )

        elapsed = int(time.time() - t0)
        names = [author_map.get(aid, {}).get("name", "?") for aid in selected_ids]
        summary = (
            f"Compared {len(selected_ids)} authors: {', '.join(names)}. "
            f"{len(unique_papers)} unique papers, {len(shared_paper_ids)} co-authored. "
            f"({elapsed}s)"
        )

        return (
            dbc.Alert(summary, color="success", className="py-2"),
            fig,
            summary,
        )

    except Exception as e:
        traceback.print_exc()
        return (
            dbc.Alert(f"Error: {type(e).__name__}: {e}", color="danger"),
            no_update,
            f"Compare failed: {e}",
        )


# ---------------------------------------------------------------------------
# Author select step 1: click candidate → preview details + Confirm button
# ---------------------------------------------------------------------------
@app.callback(
    Output("author-detail-panel", "children", allow_duplicate=True),
    Output("author-status", "children", allow_duplicate=True),
    Output("pending-author-id", "data"),
    Input({"type": "author-candidate", "id": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def author_preview_candidate(candidate_clicks):
    if not candidate_clicks or not any(c for c in candidate_clicks):
        raise PreventUpdate

    ctx = callback_context
    triggered = ctx.triggered[0] if ctx.triggered else {}
    prop_id = triggered.get("prop_id", "")
    if "author-candidate" not in prop_id:
        raise PreventUpdate

    triggered_id = json.loads(prop_id.rsplit(".", 1)[0])
    author_id = triggered_id["id"]

    details = get_author_details(author_id)
    author_info = details["author"]
    if not author_info:
        raise PreventUpdate

    papers = details["papers"]
    coauthors = details["coauthors"]

    detail_children = [
        html.H5(author_info.get("name", ""), className="mb-1"),
        html.P(author_info.get("institution", "") or "Institution unknown",
               className="text-muted small mb-1"),
        html.P([
            html.Strong(f"{author_info.get('paper_count', 0)} papers"),
            html.Span(" | ", className="mx-1"),
            html.Strong(f"{author_info.get('total_citations', 0):,} citations"),
        ], className="small mb-2"),
        dbc.Button(
            "Confirm & Build Network", id="btn-confirm-author",
            color="success", size="sm", className="mb-2 w-100",
        ),
        html.Hr(),
    ]

    if papers:
        detail_children.append(html.H6(f"Papers ({len(papers)}):", className="text-primary"))
        for p in papers[:15]:
            tier = p.get("journal_tier", "")
            tier_badge = ""
            if tier.startswith("Tier1"):
                tier_badge = "T1 "
            elif tier.startswith("Tier2"):
                tier_badge = "T2 "
            detail_children.append(html.P([
                html.Strong(f"{tier_badge}{p['title'][:70]}"),
                html.Br(),
                html.Span(
                    f"{p.get('journal', '')} ({p['year']}) — {p['citation_count']} cites",
                    className="text-muted",
                ),
            ], className="small mb-1"))
        if len(papers) > 15:
            detail_children.append(html.P(f"... and {len(papers) - 15} more",
                                          className="small text-muted"))

    if coauthors:
        detail_children.append(html.Hr())
        detail_children.append(html.H6(f"Top co-authors ({len(coauthors)}):", className="text-primary"))
        for ca in coauthors:
            detail_children.append(html.P([
                html.Strong(ca["name"]),
                html.Span(
                    f" — {ca['shared_papers']} shared papers, {ca['total_citations']:,} total cites",
                    className="text-muted",
                ),
                html.Br(),
                html.Span(ca.get("institution", "") or "", className="text-muted small"),
            ], className="small mb-1"))

    return (
        html.Div(detail_children),
        f"Preview: {author_info.get('name', '')} — click Confirm to build",
        author_id,
    )


# ---------------------------------------------------------------------------
# Author select step 2: Confirm → kick off ego network build
# ---------------------------------------------------------------------------
@app.callback(
    Output("author-detail-panel", "children", allow_duplicate=True),
    Output("author-status", "children", allow_duplicate=True),
    Output("ego-progress-interval", "disabled", allow_duplicate=True),
    Output("ego-progress-container", "style", allow_duplicate=True),
    Output("selected-author-id", "data", allow_duplicate=True),
    Input("btn-confirm-author", "n_clicks"),
    Input("author-scatter-plot", "clickData"),
    State("pending-author-id", "data"),
    State("author-landscape-ready", "data"),
    prevent_initial_call=True,
)
def author_confirm_and_build(confirm_clicks, click_data, pending_id, landscape_ready):
    ctx = callback_context
    triggered = ctx.triggered[0] if ctx.triggered else {}
    prop_id = triggered.get("prop_id", "")

    author_id = None

    if "btn-confirm-author" in prop_id and confirm_clicks and pending_id:
        author_id = pending_id
    elif "author-scatter-plot" in prop_id and click_data:
        csv_path = DATA_DIR / "authors_clustered.csv"
        if csv_path.exists():
            author_df = pd.read_csv(csv_path, encoding="utf-8-sig")
            point = click_data["points"][0]
            idx = point.get("customdata")
            if idx is not None and idx < len(author_df):
                author_id = author_df.iloc[idx]["author_id"]

    if not author_id:
        raise PreventUpdate

    details = get_author_details(author_id)
    author_info = details["author"]
    if not author_info:
        raise PreventUpdate

    # Reset progress and start background build
    _ego_progress.update({
        "percent": 0, "message": "Starting...", "running": True,
        "author_id": author_id, "result_fig": None, "result_n": 0, "done": False,
    })

    def _build_thread():
        try:
            fig, n = build_ego_network(author_id)
            _ego_progress["result_fig"] = fig
            _ego_progress["result_n"] = n
            _ego_progress["done"] = True
        except Exception as e:
            print(f"[EGO] Error: {e}")
            traceback.print_exc()
            _ego_progress["result_fig"] = go.Figure()
            _ego_progress["result_n"] = 0
            _ego_progress["done"] = True
            _ego_progress["running"] = False

    threading.Thread(target=_build_thread, daemon=True).start()

    # Reuse the detail panel from preview
    papers = details["papers"]
    coauthors = details["coauthors"]

    detail_children = [
        html.H5(author_info.get("name", ""), className="mb-1"),
        html.P(author_info.get("institution", "") or "Institution unknown",
               className="text-muted small mb-1"),
        html.P([
            html.Strong(f"{author_info.get('paper_count', 0)} papers"),
            html.Span(" | ", className="mx-1"),
            html.Strong(f"{author_info.get('total_citations', 0):,} citations"),
        ], className="small mb-2"),
        html.Hr(),
    ]

    if papers:
        detail_children.append(html.H6(f"Papers ({len(papers)}):", className="text-primary"))
        for p in papers[:15]:
            tier = p.get("journal_tier", "")
            tier_badge = ""
            if tier.startswith("Tier1"):
                tier_badge = "T1 "
            elif tier.startswith("Tier2"):
                tier_badge = "T2 "
            detail_children.append(html.P([
                html.Strong(f"{tier_badge}{p['title'][:70]}"),
                html.Br(),
                html.Span(
                    f"{p.get('journal', '')} ({p['year']}) — {p['citation_count']} cites",
                    className="text-muted",
                ),
            ], className="small mb-1"))
        if len(papers) > 15:
            detail_children.append(html.P(f"... and {len(papers) - 15} more",
                                          className="small text-muted"))

    if coauthors:
        detail_children.append(html.Hr())
        detail_children.append(html.H6(f"Top co-authors ({len(coauthors)}):", className="text-primary"))
        for ca in coauthors:
            detail_children.append(html.P([
                html.Strong(ca["name"]),
                html.Span(
                    f" — {ca['shared_papers']} shared papers, {ca['total_citations']:,} total cites",
                    className="text-muted",
                ),
                html.Br(),
                html.Span(ca.get("institution", "") or "", className="text-muted small"),
            ], className="small mb-1"))

    status = f"Building network for {author_info.get('name', '')}..."

    # Return details immediately + enable progress polling + set selected-author-id
    return html.Div(detail_children), status, False, {"display": "block"}, author_id


# ---------------------------------------------------------------------------
# Inner Circle research profile panel
# ---------------------------------------------------------------------------
def _build_inner_circle_panel() -> html.Div:
    """Build the inner circle profile section for the author detail panel."""
    ic_data = _ego_progress.get("inner_circle")
    if not ic_data or ic_data["n_members"] == 0:
        return html.Div([
            html.P("No inner circle detected.", className="text-muted small"),
        ])

    n = ic_data["n_members"]
    density = ic_data["density"]
    scores = ic_data["scores"]
    topics = ic_data["topics"]

    children = [
        html.Hr(),
        html.H6(f"Inner Circle ({n} members)", className="text-warning"),
    ]

    # Members list
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    for aid, info in ranked:
        children.append(html.P([
            html.Strong(info["name"]),
            html.Span(
                f" — {info['shared_papers']} shared, {info['total_citations']:,} cites, "
                f"score: {info['score']:.2f}",
                className="text-muted",
            ),
            html.Br(),
            html.Span(info.get("institution") or "", className="text-muted small"),
        ], className="small mb-1"))

    # Density
    density_desc = "tight team" if density >= 0.7 else "moderate cohesion" if density >= 0.4 else "hub-spoke pattern"
    children.append(html.P([
        html.Strong("Circle density: "),
        html.Span(f"{density:.2f} ({density_desc})", className="text-info"),
    ], className="small mt-2 mb-1"))

    # Topics
    if topics:
        children.append(html.P(html.Strong("Collective research focus:"), className="small mb-1"))
        for t in topics:
            children.append(html.P(
                f"  {t['topic']} ({t['count']} papers)",
                className="small mb-0 ms-2 text-muted",
            ))

    return html.Div(children)


# ---------------------------------------------------------------------------
# Progress polling: update bar + grab result when done
# ---------------------------------------------------------------------------
@app.callback(
    Output("ego-progress-bar", "value"),
    Output("ego-progress-bar", "label"),
    Output("ego-progress-label", "children"),
    Output("ego-progress-container", "style"),
    Output("ego-progress-interval", "disabled"),
    Output("author-scatter-plot", "figure", allow_duplicate=True),
    Output("author-status", "children", allow_duplicate=True),
    Output("author-detail-panel", "children", allow_duplicate=True),
    Input("ego-progress-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_ego_progress(n):
    pct = _ego_progress.get("percent", 0)
    msg = _ego_progress.get("message", "")
    done = _ego_progress.get("done", False)

    if done:
        fig = _ego_progress.get("result_fig", go.Figure())
        n_conn = _ego_progress.get("result_n", 0)
        _ego_progress["done"] = False
        _ego_progress["running"] = False

        # Build inner circle panel
        ic_panel = _build_inner_circle_panel()

        return (100, "100%", "Complete!", {"display": "none"}, True, fig,
                f"{n_conn} co-authors on map", ic_panel)

    return (pct, f"{pct}%", msg, {"display": "block"}, False, no_update, no_update,
            no_update)


def _get_filtered_df(data_path, year_range, min_citations, selected_clusters):
    df = load_dataset(Path(data_path))
    yr = year_range or [df["year"].min(), df["year"].max()]
    mask = (df["year"] >= yr[0]) & (df["year"] <= yr[1]) & (df["citation_count"] >= (min_citations or 0))
    if selected_clusters:
        mask &= df["cluster_label"].isin(selected_clusters)
    return df[mask]


# ---------------------------------------------------------------------------
# Port management
# ---------------------------------------------------------------------------
def kill_port(port: int) -> bool:
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and int(pid) != 0:
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True, timeout=5)
                    print(f"Killed previous process on port {port} (PID {pid})")
                    return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PORT = 8050

    # Kill anything on our port first (most reliable method on Windows)
    kill_port(PORT)

    # Also try PID file as backup
    pid_file = DATA_DIR / "app.pid"
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if old_pid != os.getpid():
                subprocess.run(["taskkill", "/F", "/PID", str(old_pid)],
                               capture_output=True, timeout=5)
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)

    time.sleep(0.5)
    pid_file.write_text(str(os.getpid()))

    # ---- Auto-setup: build missing components ----
    APP_VERSION = "v4"
    print(f"=== Research Landscape Map {APP_VERSION} ===")

    db_path = DATA_DIR / "openalex.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        print(f"  Database: {papers:,} papers")

        # Auto-build FTS5 if missing
        if "papers_fts" not in tables:
            print("  FTS5 index missing, building now (this may take a few minutes)...")
            from build_db import build_fts5_index
            build_fts5_index()
            print("  FTS5 index: ready")
        else:
            print("  FTS5 index: ready")

        # Auto-build author tables if missing
        has_authors = "authors" in tables
        author_count = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0] if has_authors else 0
        conn.close()

        if not has_authors or author_count == 0:
            # Check if snapshot is available
            from download_openalex import WORKS_DIR
            if WORKS_DIR.exists() and any(WORKS_DIR.rglob("*.gz")):
                print("  Author tables missing, building from snapshot (this will take a while)...")
                from build_db import build_authors
                build_authors()
                print("  Author tables: ready")
            else:
                print("  Author tables: skipped (snapshot not found at N:/openalex-snapshot/)")
        else:
            print(f"  Author tables: {author_count:,} authors")
    else:
        print("  WARNING: No database at data/openalex.db")

    if datasets:
        print(f"  Datasets: {list(datasets.keys())}")
    print(f"\n  http://localhost:{PORT}")
    print()
    app.run(debug=False, host="0.0.0.0", port=PORT)
