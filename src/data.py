from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# Load project configuration (paths etc.)
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

with open(PROJECT_ROOT / "configs" / "default.yml") as f:
    _config = yaml.safe_load(f)

RAW_FILE = PROJECT_ROOT / _config["paths"]["raw_data"]
PROCESSED_DIR = PROJECT_ROOT / _config["paths"]["processed_data"]


# -------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------
def load_raw(path: str | Path = RAW_FILE) -> pd.DataFrame:
    """Load raw dataset as DataFrame."""
    path = Path(path)
    return pd.read_csv(path)

def save_df(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to CSV (create dirs if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -------------------------------------------------------------------
# Dataset splitting
# -------------------------------------------------------------------
def split_and_save(
    df: pd.DataFrame,
    out_dir: str | Path = PROCESSED_DIR,
    test_size: float = 0.2,
    stratify_col: str | None = "Churn",
    random_state: int = 42,
    min_examples_per_class_in_test: int = 50
) -> None:
    """
    Stratified train/test split only (no validation set).
    Saves train.csv and test.csv under out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide strat array (or None)
    strat = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    # Safety check: estimate minority examples in test and adjust test_size if too small
    if strat is not None:
        counts = strat.value_counts()
        minority_frac = counts.min() / len(df)
        est_minority_in_test = int(len(df) * test_size * minority_frac)
        if est_minority_in_test < min_examples_per_class_in_test:
            needed_test_size = min(0.5, min_examples_per_class_in_test / (len(df) * minority_frac))
            if needed_test_size > test_size:
                print(f"[warning] estimated minority examples in test ({est_minority_in_test}) < "
                      f"{min_examples_per_class_in_test}. Increasing test_size {test_size:.3f} -> {needed_test_size:.3f}")
                test_size = needed_test_size

    # Perform stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )

    # Save results
    save_df(train_df, out_dir / "train.csv")
    save_df(test_df, out_dir / "test.csv")
    print(f"Saved train ({len(train_df)}) and test ({len(test_df)}) to {out_dir}. "
          f"Final test_size used: {len(test_df)/len(df):.3f} (requested approx {test_size:.3f}).")
