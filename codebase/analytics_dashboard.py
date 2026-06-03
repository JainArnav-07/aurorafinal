import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"

SEGMENTS_FILE = OUTPUT_DIR / "users_segmented.csv"
SCHEDULE_FILE = OUTPUT_DIR / "user_schedules_final.csv"
FREQUENCY_FILE = OUTPUT_DIR / "users_frequency_assigned.csv"
TIMING_FILE = OUTPUT_DIR / "timing_recommendations.csv"

# --------------------------------------------------
# HEADER
# --------------------------------------------------

print("=" * 70)
print("AURORA ANALYTICS DASHBOARD")
print("=" * 70)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

segments_df = pd.read_csv(SEGMENTS_FILE)
schedule_df = pd.read_csv(SCHEDULE_FILE)
frequency_df = pd.read_csv(FREQUENCY_FILE)
timing_df = pd.read_csv(TIMING_FILE)

print("\nData Loaded Successfully")
print(f"Users Loaded: {len(segments_df)}")

# --------------------------------------------------
# SEGMENT DISTRIBUTION
# --------------------------------------------------

print("\n" + "=" * 70)
print("SEGMENT DISTRIBUTION")
print("=" * 70)

segment_col = None

for col in segments_df.columns:
    if "segment" in col.lower():
        segment_col = col
        break

if segment_col:

    segment_counts = segments_df[segment_col].value_counts()

    for seg, count in segment_counts.items():

        pct = round((count / len(segments_df)) * 100, 2)

        print(f"{seg}: {count} users ({pct}%)")

# --------------------------------------------------
# ACTIVATION SCORE ANALYSIS
# --------------------------------------------------

activation_col = None

for col in segments_df.columns:
    if "activation" in col.lower():
        activation_col = col
        break

if activation_col:

    print("\n" + "=" * 70)
    print("ACTIVATION SCORE ANALYSIS")
    print("=" * 70)

    print(
        f"Average Activation Score: "
        f"{segments_df[activation_col].mean():.3f}"
    )

    print(
        f"Maximum Activation Score: "
        f"{segments_df[activation_col].max():.3f}"
    )

    print(
        f"Minimum Activation Score: "
        f"{segments_df[activation_col].min():.3f}"
    )

# --------------------------------------------------
# FREQUENCY ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 70)
print("NOTIFICATION FREQUENCY ANALYSIS")
print("=" * 70)

for col in frequency_df.columns:

    if (
        "notif" in col.lower()
        or "frequency" in col.lower()
        or "count" in col.lower()
    ):

        print(f"\nColumn: {col}")
        print(frequency_df[col].value_counts().head())

# --------------------------------------------------
# TIMING ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 70)
print("TIMING RECOMMENDATION ANALYSIS")
print("=" * 70)

print(f"Timing Recommendation Rows: {len(timing_df)}")

print("\nFirst 5 Timing Recommendations:")

print(timing_df.head())

# --------------------------------------------------
# SCHEDULE ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 70)
print("SCHEDULE ANALYSIS")
print("=" * 70)

print(f"Scheduled Users: {schedule_df.shape[0]}")
print(f"Schedule Columns: {schedule_df.shape[1]}")

slot_cols = []

for col in schedule_df.columns:

    if (
        "slot" in col.lower()
        or "time" in col.lower()
        or "notif" in col.lower()
    ):
        slot_cols.append(col)

print(f"Detected Schedule Columns: {len(slot_cols)}")

# --------------------------------------------------
# SUMMARY EXPORT
# --------------------------------------------------

total_segments = (
    segments_df[segment_col].nunique()
    if segment_col
    else 0
)

summary = pd.DataFrame(
    {
        "metric": [
            "total_users",
            "total_segments",
            "scheduled_users",
            "timing_recommendations"
        ],
        "value": [
            len(segments_df),
            total_segments,
            schedule_df.shape[0],
            len(timing_df)
        ]
    }
)

summary_path = OUTPUT_DIR / "analytics_summary.csv"

summary.to_csv(summary_path, index=False)

print("\n" + "=" * 70)
print("EXPORT SUCCESSFUL")
print("=" * 70)

print(f"Saved: {summary_path}")

print("\nAnalytics Dashboard Completed Successfully")