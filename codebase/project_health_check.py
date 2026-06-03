import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"

segments_df = pd.read_csv(OUTPUT_DIR / "users_segmented.csv")
timing_df = pd.read_csv(OUTPUT_DIR / "timing_recommendations.csv")
frequency_df = pd.read_csv(OUTPUT_DIR / "users_frequency_assigned.csv")

# -------------------------------
# Activation Score Insights
# -------------------------------

activation_col = [
    c for c in segments_df.columns
    if "activation" in c.lower()
][0]

avg_activation = round(
    segments_df[activation_col].mean(),
    3
)

max_activation = round(
    segments_df[activation_col].max(),
    3
)

min_activation = round(
    segments_df[activation_col].min(),
    3
)

# -------------------------------
# Segment Insights
# -------------------------------

segment_col = [
    c for c in segments_df.columns
    if "segment" in c.lower()
][0]

largest_segment = (
    segments_df[segment_col]
    .value_counts()
    .idxmax()
)

largest_segment_count = (
    segments_df[segment_col]
    .value_counts()
    .max()
)

# -------------------------------
# Timing Insights
# -------------------------------

best_window = (
    timing_df.groupby("optimal_window")
    ["expected_ctr"]
    .mean()
    .idxmax()
)

best_ctr = round(
    timing_df.groupby("optimal_window")
    ["expected_ctr"]
    .mean()
    .max(),
    3
)

# -------------------------------
# Notification Frequency Insights
# -------------------------------

avg_daily_limit = round(
    frequency_df["daily_notification_limit"].mean(),
    2
)

# -------------------------------
# Report
# -------------------------------

report = pd.DataFrame({
    "metric": [
        "avg_activation_score",
        "max_activation_score",
        "min_activation_score",
        "largest_segment",
        "largest_segment_users",
        "best_time_window",
        "best_window_ctr",
        "avg_daily_notification_limit"
    ],
    "value": [
        avg_activation,
        max_activation,
        min_activation,
        largest_segment,
        largest_segment_count,
        best_window,
        best_ctr,
        avg_daily_limit
    ]
})

output_file = OUTPUT_DIR / "project_health_report.csv"

report.to_csv(output_file, index=False)

print("=" * 60)
print("PROJECT HEALTH REPORT GENERATED")
print("=" * 60)
print(report)
print(f"\nSaved to: {output_file}")