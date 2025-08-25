import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def compute_rmssd(rr_intervals):
    """
    Compute RMSSD (Root Mean Square of Successive Differences) from RR intervals.

    Args:
        rr_intervals (list or np.ndarray): List of RR intervals in milliseconds.

    Returns:
        float: RMSSD value in milliseconds.
    """
    rr_intervals = np.array(rr_intervals, dtype=float)

    # Compute successive differences
    diff = np.diff(rr_intervals)

    # Square the differences
    squared_diff = diff ** 2

    # Compute mean and square root
    rmssd = np.sqrt(np.mean(squared_diff))

    return rmssd


def compute_all_rmssd(corrected_data):
    """
    Apply calculation to all participants and phases.

    Args:
        corrected_data (dict): Nested dictionary with corrected RR data and labels.

    Returns:
        list of dicts: Each dict contains participant ID and RMSSD for each phase.
    """
    results = []
    for pid, phases in corrected_data.items():
        row = {"ID": pid}
        for phase in ["baseline", "task", "recovery"]:
            rr = phases[phase]["corrected_rr"]
            row[phase] = compute_rmssd(rr)
        results.append(row)
    return results


def save_results(results, results_path, file_name):
    """
    Save RMSSD results to an Excel file.

    Args:
        results (list of dict): Output from compute_all_rmssd().
        results_path (str): Path where the Excel file will be saved.
    """
    df = pd.DataFrame(results)
    df.set_index("ID", inplace=True)
    save_path = os.path.join(results_path, file_name)
    df.to_excel(save_path, index=True)
    print(f"Results saved to {save_path}")

    return df


def annotate_outliers(ax, data, label_x):
    """
    Annotate outliers in a boxplot by participant index.

    Args:
        ax (matplotlib.axes.Axes): The axes to annotate.
        data (pd.Series): The data to check for outliers.
        label_x (int): The x-axis position of the box in the plot.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    for idx, y_val in outliers.items():
        ax.annotate(
            str(idx),
            xy=(label_x, y_val),
            xytext=(label_x + 0.05, y_val),
            ha='left', va='center',
            fontsize=8,
            color='red'
        )


def create_boxplots(rmssd_df, save_path, file_name):
    """
    Create a boxplot of RMSSD values across stages (baseline, task, recovery).

    Args:
        rmssd_df (pd.DataFrame): DataFrame containing RMSSD values per participant.
                                 Must have columns: 'baseline', 'task', 'recovery'.
        save_path (str): Path to save the plot to.
    """
    # Extract values for each stage, removing NaNs
    stages = ['baseline', 'task', 'recovery']
    data_to_plot = [rmssd_df[stage].dropna() for stage in stages]

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create the boxplot
    ax.boxplot(
        data_to_plot,
        labels=[s.capitalize() for s in stages],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='black'),
        medianprops=dict(color='darkblue'),
        whiskerprops=dict(color='gray'),
        capprops=dict(color='gray')
    )

    # Annotate outliers for each stage
    for i, stage in enumerate(stages):
        annotate_outliers(ax, rmssd_df[stage].dropna(), label_x=i + 1)

    ax.set_title("RMSSD per Stage")
    ax.set_ylabel("RMSSD (ms)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(save_path, file_name)

    plt.savefig(save_path)
    plt.close()
    print(f"Boxplot saved to {save_path}")


def remove_outliers(rmssd_df, lower_threshold=10, upper_threshold=120):
    """
    Remove RMSSD values that are outliers:
    - Below lower threshold
    - Above upper threshold
    - Outside 1.5 * IQR bounds

    Args:
        rmssd_df (pd.DataFrame): DataFrame with columns ['baseline', 'task', 'recovery']
        lower_threshold (float): Minimum plausible RMSSD value (inclusive)
        upper_threshold (float): Maximum plausible RMSSD value (inclusive)

    Returns:
        pd.DataFrame: Cleaned DataFrame with outliers replaced by NaN
    """
    cleaned_df = rmssd_df.copy()

    for stage in ['baseline', 'task', 'recovery']:
        data = cleaned_df[stage]

        # Drop NaNs before calculating IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR

        # Create boolean mask for valid values
        is_valid = (
            (data >= lower_threshold) &
            (data <= upper_threshold) &
            (data >= iqr_lower) &
            (data <= iqr_upper)
        )

        # Set invalid (outlier) values to NaN
        cleaned_df[stage] = data.where(is_valid)

    return cleaned_df
