import re
import glob
import os
import numpy as np
import pandas as pd

# Configuration
GUIDE_MAP_PATTERN = "*GuideMap.csv"
CELL_LINE_PATTERN = re.compile(
    r"^(?:Genomic_screen_)?([A-Za-z0-9\-]+)_(?:end|Final)", re.IGNORECASE
)
MIN_MEASUREMENTS = 2
NORM_FLOOR = 1e-3
OUTPUT_NORM_CSV = "depmap_empirical_sd_norm.csv"


def parse_cell_line(column_name):
    m = CELL_LINE_PATTERN.match(column_name)
    return m.group(1) if m else None


def load_guide_map(guide_map_path):
    gm = pd.read_csv(guide_map_path, low_memory=False)
    # Restrict to Chronos-used guides if flag exists
    if "UsedByChronos" in gm.columns:
        gm = gm[gm["UsedByChronos"] == True]

    # Use regex=False so ' (' is treated as a literal string, not an unclosed regex group
    gm["gene_symbol"] = gm["Gene"].astype(str).str.split(" (", n=1, regex=False).str[0]
    gm = gm.drop_duplicates(subset="sgRNA", keep="first")
    return dict(zip(gm["sgRNA"], gm["gene_symbol"]))


def process_single_screen(guide_map_path, lfc_path):
    sgrna_to_gene = load_guide_map(guide_map_path)
    lfc = pd.read_csv(lfc_path, index_col=0, low_memory=False)

    col_to_cell = {c: parse_cell_line(c) for c in lfc.columns if parse_cell_line(c)}
    if not col_to_cell:
        return None

    lfc = lfc[list(col_to_cell.keys())]
    lfc = lfc.reset_index().rename(columns={lfc.index.name or "index": "sgRNA"})

    long_df = lfc.melt(id_vars="sgRNA", var_name="screen_col", value_name="lfc")
    long_df["gene"] = long_df["sgRNA"].map(sgrna_to_gene)
    long_df["cell_line"] = long_df["screen_col"].map(col_to_cell)

    long_df = long_df.dropna(subset=["gene", "cell_line", "lfc"])

    # Aggregate within screen
    grp = long_df.groupby(["gene", "cell_line"])["lfc"]
    stats = grp.agg(n="count", var="var", mean="mean").reset_index()
    stats["var"] = stats["var"].fillna(0.0)
    return stats


def build_consolidated_norm_matrix(data_dir="."):
    gmaps = sorted(glob.glob(os.path.join(data_dir, GUIDE_MAP_PATTERN)))
    all_stats = []

    for gmap in gmaps:
        # Match corresponding LogfoldChange file
        prefix = gmap.replace("GuideMap.csv", "")
        lfc_path = f"{prefix}LogfoldChange.csv"
        if not os.path.exists(lfc_path):
            continue

        print(f"Processing screen pair: {gmap} & {lfc_path}")
        stats = process_single_screen(gmap, lfc_path)
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        raise FileNotFoundError("No matching GuideMap / LogfoldChange pairs found.")

    combined = pd.concat(all_stats, ignore_index=True)

    # Pool variances across libraries/screens if overlapping (gene, cell_line) exist
    def pool_group(df):
        N = df["n"].sum()
        if N < 2:
            return pd.Series({"n_total": N, "pooled_sd": NORM_FLOOR})
        # Pooled variance formula
        ss_within = (df["n"] - 1).clip(lower=0) * df["var"]
        pooled_var = ss_within.sum() / max(N - 1, 1)
        sd = np.sqrt(pooled_var)
        return pd.Series({"n_total": N, "pooled_sd": max(sd, NORM_FLOOR)})

    print("Pooling empirical variances across screens...")
    pooled = combined.groupby(["gene", "cell_line"]).apply(pool_group, include_groups=False).reset_index()

    # Pivot into Wide Matrix (genes x cell_lines)
    wide_norm = pooled.pivot(index="gene", columns="cell_line", values="pooled_sd")
    wide_norm.to_csv(OUTPUT_NORM_CSV)
    print(f"Consolidated norm matrix saved to '{OUTPUT_NORM_CSV}'. Shape: {wide_norm.shape}")
    return wide_norm


if __name__ == "__main__":
    build_consolidated_norm_matrix()
