from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOTS_DIR = Path(__file__).parents[2] / "data" / "snapshots"

A = SNAPSHOTS_DIR / "2026-04_35581rows.csv"
B = SNAPSHOTS_DIR / "2026-04_32047rows.csv"


def main() -> None:
    df_a = pd.read_csv(A, encoding="utf-8-sig")
    df_b = pd.read_csv(B, encoding="utf-8-sig")

    print(f"Snapshot A ({A.name}): {len(df_a):>6} rows")
    print(f"Snapshot B ({B.name}): {len(df_b):>6} rows")

    overlap = df_a["url"].isin(df_b["url"]).sum()
    print(f"\nOverlap (same URL in both): {overlap:>6} rows")
    print(f"Unique to A:                {len(df_a) - overlap:>6} rows")
    print(f"Unique to B:                {len(df_b) - df_b['url'].isin(df_a['url']).sum():>6} rows")

    merged = (
        pd.concat([df_a, df_b], ignore_index=True)
        .drop_duplicates(subset=["url"])
        .reset_index(drop=True)
    )
    print(f"\nMerged (unique):            {len(merged):>6} rows")

    print("\n--- Records per city ---")
    city_counts = merged["city"].value_counts()
    for city, n in city_counts.items():
        print(f"  {city:<15} {n:>6}")
    print(f"  {'TOTAL':<15} {len(merged):>6}")

    print("\n--- Records per city + neighborhood (top 5 per city) ---")
    for city in city_counts.index:
        city_df = merged[merged["city"] == city]
        neigh_counts = city_df["neighborhood"].value_counts().head(5)
        print(f"\n  {city}")
        for neigh, n in neigh_counts.items():
            print(f"    {neigh:<35} {n:>5}")

    out_path = SNAPSHOTS_DIR / f"2026-04_merged_{len(merged)}rows.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nMerged file saved → {out_path}")


if __name__ == "__main__":
    main()
