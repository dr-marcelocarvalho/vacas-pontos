"""
analyze_features.py — Análise Estatística Descritiva.

Carrega data/processed/features.csv e gera:
- Histogramas das features
- Heatmap de correlação
- Boxplots por station_id e camera_id
- Pairplot das features principais

Salva PNGs e um resumo descritivo em outputs/figures/.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core_utils import DATA_DIR, OUTPUTS_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

# Estilo
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({"figure.max_open_warning": 0})


def main() -> None:
    print("=" * 60)
    print("  🐄 ANÁLISE DESCRITIVA — Cows Challenge")
    print("=" * 60)

    csv_path = DATA_DIR / "processed" / "features.csv"
    if not csv_path.exists():
        print(f"  ❌ CSV não encontrado: {csv_path}")
        print("  Execute extract_features.py primeiro.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  📊 Shape: {df.shape}")

    figures_dir = OUTPUTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Separar colunas numéricas de features
    angle_cols = [c for c in df.columns if c.startswith("angle_")]
    dist_cols = [c for c in df.columns if c.startswith("dist_")]
    ratio_cols = [c for c in df.columns if c.startswith("ratio_")]
    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    feature_cols = angle_cols + dist_cols + ratio_cols

    print(f"  📐 Ângulos: {len(angle_cols)}")
    print(f"  📏 Distâncias: {len(dist_cols)}")
    print(f"  ⚖️  Proporções: {len(ratio_cols)}")
    print()

    # ─── 1. Histogramas ───
    print("  📊 Gerando histogramas...")
    n_features = len(feature_cols)
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=6)

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Distribuição das Features Geométricas", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(figures_dir / "histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── 2. Heatmap de Correlação ───
    print("  🌡️  Gerando heatmap de correlação...")
    if len(feature_cols) > 0:
        corr = df[feature_cols].corr()
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            corr,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_title("Correlação entre Features Geométricas", fontsize=14)
        ax.tick_params(labelsize=6)
        plt.tight_layout()
        fig.savefig(figures_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ─── 3. Boxplots por Station ───
    print("  📦 Gerando boxplots por station...")
    station_col = "meta_station"
    if station_col in df.columns and df[station_col].notna().sum() > 0:
        top_features = (angle_cols + ratio_cols)[:6]
        if top_features:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            for i, col in enumerate(top_features):
                if i >= len(axes):
                    break
                sns.boxplot(data=df, x=station_col, y=col, ax=axes[i], palette="Set2")
                axes[i].set_title(col, fontsize=9)
                axes[i].tick_params(axis="x", rotation=45, labelsize=6)
                axes[i].tick_params(axis="y", labelsize=7)

            for i in range(len(top_features), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle("Distribuição por Estação (Station)", fontsize=14)
            plt.tight_layout()
            fig.savefig(figures_dir / "boxplot_by_station.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ─── 4. Boxplots por Camera ───
    print("  📷 Gerando boxplots por câmera...")
    cam_col = "meta_cam"
    if cam_col in df.columns and df[cam_col].notna().sum() > 0:
        top_features_cam = (angle_cols + ratio_cols)[:6]
        if top_features_cam:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            for i, col in enumerate(top_features_cam):
                if i >= len(axes):
                    break
                sns.boxplot(data=df, x=cam_col, y=col, ax=axes[i], palette="Set3")
                axes[i].set_title(col, fontsize=9)
                axes[i].tick_params(axis="x", rotation=45, labelsize=6)
                axes[i].tick_params(axis="y", labelsize=7)

            for i in range(len(top_features_cam), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle("Distribuição por Câmera", fontsize=14)
            plt.tight_layout()
            fig.savefig(figures_dir / "boxplot_by_camera.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ─── 5. Pairplot (amostra) ───
    print("  🔗 Gerando pairplot...")
    pairplot_cols = angle_cols[:3] + ratio_cols[:3]
    if len(pairplot_cols) >= 2:
        sample_df = df[pairplot_cols].dropna().sample(
            n=min(500, len(df)), random_state=42
        )
        g = sns.pairplot(sample_df, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 10})
        g.fig.suptitle("Pairplot — Features Principais", y=1.02, fontsize=14)
        g.savefig(figures_dir / "pairplot.png", dpi=120)
        plt.close("all")

    # ─── 6. Resumo Descritivo ───
    print("  📝 Gerando resumo descritivo...")
    desc = df[feature_cols].describe().T

    md_lines = [
        "# Análise Descritiva — Features Geométricas\n",
        f"**Total de amostras:** {len(df)}\n",
        f"**Total de features:** {len(feature_cols)}\n",
        "",
        "## Estatísticas Descritivas\n",
        desc.to_markdown(),
        "",
        "## Figuras Geradas\n",
        "- `histograms.png` — Distribuição de cada feature",
        "- `correlation_heatmap.png` — Correlação entre features",
        "- `boxplot_by_station.png` — Variação por estação",
        "- `boxplot_by_camera.png` — Variação por câmera",
        "- `pairplot.png` — Relações entre features principais",
    ]

    desc_path = figures_dir / "descriptive_analysis.md"
    desc_path.write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print("=" * 60)
    print("  ✅ ANÁLISE COMPLETA")
    print(f"  📁 Figuras em: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
