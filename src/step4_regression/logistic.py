"""
Step 4: ロジスティック回帰

    log(p / (1-p)) = β0 + β1·TypeA + β2·TypeB + β3·Trend + β4·Quality

β1が有意 → 品質と独立に、陣営反応でノートが潰されている証拠。
"""

import pandas as pd
import statsmodels.api as sm


def fit_logistic_regression(df: pd.DataFrame):
    """
    ロジスティック回帰を実行する。

    Parameters
    ----------
    df : pd.DataFrame
        columns: deleted, type_a, type_b, trend, quality

    Returns
    -------
    result : statsmodels GLMResultsWrapper
    """
    y = df["deleted"]
    X = df[["type_a", "type_b", "trend", "quality"]]
    X = sm.add_constant(X)

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    print("\n" + "=" * 60)
    print("  Logistic Regression Results")
    print("=" * 60)
    summary = result.summary2().tables[1]
    for idx, row in summary.iterrows():
        coef = row["Coef."]
        pval = row["P>|z|"]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {idx:<12} β={coef:+.4f}  p={pval:.4f} {sig}")

    print()
    beta1_p = summary.loc["type_a", "P>|z|"] if "type_a" in summary.index else 1.0
    if beta1_p < 0.05:
        print("  → β1(TypeA) is significant: faction reaction affects note status")
    else:
        print("  → β1(TypeA) is NOT significant")
    print("=" * 60)

    return result
