import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


config = {
    # CSV file path
    "csv_file": Path('data/20250305/imagedata10/data.csv'),

    # Output directory path (for storing filtered CSV)
    "output_folder": Path('data/20250305/imagedata10'),

    # Enable time range filtering (True/False)
    "enable_time_filter": False,

    # Time range filters (leave empty for all time periods)
    "time_ranges": [
        {"lower": datetime( ,  ,  ,  ,  ,  ), "upper": datetime( ,  ,  ,  ,  ,  )}
    ],

    # Exclusion period after flow rate changes
    "exclude_duration": timedelta(seconds= ),

    # Exclude data at start and end of selected range
    "exclude_start_end": True,
    "start_exclude_duration": timedelta(seconds= ),  # Start period
    "end_exclude_duration": timedelta(seconds= )  # End period
}


def apply_time_filters(df: pd.DataFrame, time_ranges: list, enable_filter: bool) -> pd.DataFrame:

    if not enable_filter or not time_ranges:
        logger.info("Time filtering disabled, keeping all data.")
        return df

    mask = pd.Series(False, index=df.index)
    for trange in time_ranges:
        lower = trange.get("lower")
        upper = trange.get("upper")

        current_mask = pd.Series(True, index=df.index)
        if lower:
            current_mask &= (df['timestamp'] >= lower)
        if upper:
            current_mask &= (df['timestamp'] <= upper)

        mask |= current_mask

    return df[mask].copy()


def exclude_start_end_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:

    if not config.get("exclude_start_end", False):
        return df

    start_exclude_duration = config.get("start_exclude_duration", timedelta(seconds=0))
    end_exclude_duration = config.get("end_exclude_duration", timedelta(seconds=0))

    if df.empty:
        return df

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()

    # Exclude start period
    if start_exclude_duration.total_seconds() > 0:
        start_cut_off = min_time + start_exclude_duration
        df = df[df['timestamp'] > start_cut_off]

    # Exclude end period
    if end_exclude_duration.total_seconds() > 0 and not df.empty:
        max_time = df['timestamp'].max()
        end_cut_off = max_time - end_exclude_duration
        df = df[df['timestamp'] < end_cut_off]

    return df.reset_index(drop=True)


def main(config):
    csv_file = config["csv_file"]
    output_folder = config["output_folder"]
    exclude_duration = config["exclude_duration"]
    enable_time_filter = config.get("enable_time_filter", False)

    # Create output directory if needed
    output_folder.mkdir(parents=True, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Apply time range filtering (optional)
    df_filtered = apply_time_filters(df, config.get("time_ranges", []), enable_time_filter)

    # Sort data by timestamp
    df_filtered = df_filtered.sort_values(by='timestamp').reset_index(drop=True)

    # Exclude start/end periods
    df_filtered = exclude_start_end_data(df_filtered, config)

    if df_filtered.empty:
        logger.warning("No data remained after filtering.")
        (output_folder / 'filtered_images.csv').write_text("No data after filtering.")
        return


    df_filtered['flow_rate_diff'] = df_filtered['flow_rate'].diff().ne(0)

    exclude_indices = set()
    exclude_until = None

    for i in tqdm(df_filtered.index, desc="Filtering data", total=df_filtered.shape[0]):
        if i == 0:
            continue

        current_time = df_filtered.at[i, 'timestamp']
        previous_time = df_filtered.at[i - 1, 'timestamp']
        flow_rate_changed = df_filtered.at[i, 'flow_rate_diff']

        time_diff = (current_time - previous_time).total_seconds()

        if flow_rate_changed:
            if time_diff < exclude_duration.total_seconds():
                exclude_indices.update([i, i - 1])
            exclude_until = current_time + exclude_duration

        if exclude_until and current_time <= exclude_until:
            exclude_indices.add(i)

    df_final = df_filtered.drop(index=exclude_indices).reset_index(drop=True)

    target_csv_file = output_folder / 'filtered_images.csv'
    df_final.to_csv(target_csv_file, index=False)

    logger.info("Processing complete! Filtered CSV saved to output folder.")


if __name__ == "__main__":
    main(config)