import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')


class RRArtifactCorrector:
    """
    Initialize the artifact corrector.

    Args:
        scaling_factor (float): Scaling factor for threshold calculation. Default is 5.2.
        window_size (int): Size of rolling window for threshold calculation. Default is 91 beats.
    """

    def __init__(self, scaling_factor=5.2, window_size=91):
        self.scaling_factor = scaling_factor
        self.window_size = window_size

    def load_data(self, folder_path):
        """
        Load RR interval data from text files organized in folders.

        Args:
            folder_path (str): Path to the folder containing participant subfolders.

        Returns:
            dict: Nested dictionary {participant_id: {phase: rr_intervals_list}}.
        """
        participants_paths = sorted([os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path)])
        data = {}

        for pp in participants_paths:
            files_paths = sorted([os.path.join(pp, file) for file in os.listdir(pp)])
            participant_id = os.path.basename(pp)

            baseline = pd.read_csv(files_paths[0], header=None).squeeze("columns").tolist()
            recovery = pd.read_csv(files_paths[1], header=None).squeeze("columns").tolist()
            task = pd.read_csv(files_paths[2], header=None).squeeze("columns").tolist()

            data[participant_id] = {
                'baseline': baseline,
                'task': task,
                'recovery': recovery
            }
        return data

    def calculate_dRRs(self, rr_intervals):
        """
        Calculate differences between successive RR intervals (dRRs).

        Args:
            rr_intervals (list or array): List of RR intervals.

        Returns:
            np.ndarray: Array of dRR values.
        """
        dRRs = np.diff(rr_intervals, prepend=rr_intervals[0])
        return dRRs

    def calculate_mRRs(self, rr_intervals):
        """
        Calculate difference between each RR interval and local 11-beat median (mRRs).

        Args:
            rr_intervals (list or array): List of RR intervals.

        Returns:
            np.ndarray: Array of mRR values after scaling.
        """
        medians = pd.Series(rr_intervals).rolling(window=11, center=True, min_periods=1).median()
        mRRs = np.array(rr_intervals) - medians
        mRRs = np.where(mRRs < 0, 2 * mRRs, mRRs)
        return mRRs

    def calculate_threshold(self, series):
        """
        Calculate a time-varying threshold based on quartile deviation.

        Args:
            series (list or array): Series to calculate threshold on (dRRs or mRRs).

        Returns:
            np.ndarray: Threshold values.
        """
        series = pd.Series(series)
        q1 = series.rolling(window=self.window_size, center=True, min_periods=1).quantile(0.25)
        q3 = series.rolling(window=self.window_size, center=True, min_periods=1).quantile(0.75)
        qd = (q3 - q1) / 2
        threshold = self.scaling_factor * qd
        threshold = threshold.fillna(method='bfill').fillna(method='ffill')
        return threshold.to_numpy()

    def normalize_series(self, series, threshold):
        """
        Normalize a series by its dynamic threshold.

        Args:
            series (np.ndarray): Input series (dRRs or mRRs).
            threshold (np.ndarray): Corresponding threshold.

        Returns:
            np.ndarray: Normalized series.
        """
        return series / threshold

    def calculate_S1(self, dRR_norm):
        """
        Calculate S1 space used for ectopic beat detection.
        Ectopic Beats are beats that come too early or too late, causing a sharp change in RR interval,
        and are followed by a "correction beat" (compensating correction).

        Args:
            dRR_norm (np.ndarray): Normalized dRR series.

        Returns:
            tuple: S11 (current dRR) and S12 (pattern of surrounding dRRs).
        """
        S11 = dRR_norm  # The normalized change at one beat relative to its previous one
        S12 = np.zeros_like(dRR_norm)  # Initialize S12, same length as dRR

        for j in range(len(dRR_norm)):  # Iterate over dRR

            # For a positive change: look at surrounding values and take the larger one (and not the "correction beat")
            if dRR_norm[j] > 0:
                S12[j] = max(dRR_norm[j - 1] if j - 1 >= 0 else 0, dRR_norm[j + 1] if j + 1 < len(dRR_norm) else 0)

            # For a negative change: take the smaller surrounding value (and not the "correction beat")
            else:
                S12[j] = min(dRR_norm[j - 1] if j - 1 >= 0 else 0, dRR_norm[j + 1] if j + 1 < len(dRR_norm) else 0)

        return S11, S12

    def calculate_S2(self, dRR_norm):
        """
        Calculate S2 space used for missed/extra beat detection,
        that create a distortion across several beats.

        Args:
            dRR_norm (np.ndarray): Normalized dRR series.

        Returns:
            tuple: S21 (current dRR) and S22 (next 2 dRRs compared).
        """
        S21 = dRR_norm  # dRR (like in S1)
        S22 = np.zeros_like(dRR_norm)  # either the min or max of the next two dRRs (depending on direction)

        for j in range(len(dRR_norm)):  # Iterate over dRR

            # If current dRR is positive or zero, check for missed beats
            if dRR_norm[j] >= 0:
                S22[j] = min(
                    dRR_norm[j + 1] if j + 1 < len(dRR_norm) else 0,
                    dRR_norm[j + 2] if j + 2 < len(dRR_norm) else 0
                )

            # If negative, check for extra beats
            else:
                S22[j] = max(
                    dRR_norm[j + 1] if j + 1 < len(dRR_norm) else 0,
                    dRR_norm[j + 2] if j + 2 < len(dRR_norm) else 0
                )

        return S21, S22

    def classify_beats(self, dRR_norm, mRR_norm, rr_intervals, Th2):
        """
        Classify each beat as 'normal', 'ectopic', 'long/short', 'missed', or 'extra'.

        Args:
            dRR_norm (np.ndarray): Normalized dRR series.
            mRR_norm (np.ndarray): Normalized mRR series.
            rr_intervals (list): Original RR interval series.
            Th2 (np.ndarray): Threshold values.

        Returns:
            list: Beat labels for each interval.
        """
        # Calculate S1 and S2 spaces
        S11, S12 = self.calculate_S1(dRR_norm)
        S21, S22 = self.calculate_S2(dRR_norm)

        # Get the non-normalized mRR back
        mRR = mRR_norm * Th2

        # Threshold constants from the paper
        c1, c2 = 0.13, 0.17

        labels = ["normal"] * len(dRR_norm)  # Default all beats to normal

        for j in range(len(dRR_norm)):  # Iterate over dRR_norm

            # Detect ectopic beats using S1 (sharp spike or dip pattern)
            if (S11[j] > 1 and S12[j] < -c1 * S11[j] - c2) or (S11[j] < -1 and S12[j] > -c1 * S11[j] + c2):
                labels[j] = "ectopic"

            # Detect long or short beats using S2 and mRR deviation
            elif (S21[j] > 1 and S22[j] < -1) or (S21[j] < -1 and S22[j] > 1) or abs(mRR_norm[j]) > 3:
                labels[j] = "long/short"

                # Detect missed or extra beat
                if abs(rr_intervals[j]/2 - mRR[j]) < Th2[j]:
                    labels[j] = "missed"
                elif j < len(rr_intervals) - 1 and abs(rr_intervals[j] + rr_intervals[j+1] - mRR[j]) < Th2[j]:
                    labels[j] = "extra"

        return labels

    def correct_rr_intervals(self, rr_intervals, labels):
        """
        Correct RR intervals based on artifact labels.

        Args:
            rr_intervals (list): Original RR intervals.
            labels (list): Labels for each interval ("normal", "extra", "missed", "ectopic", "long/short").

        Returns:
            rr_corrected (list): Corrected RR interval series.
            corrected_labels (list):
        """
        rr_corrected = list(rr_intervals)  # Use list to allow insert/delete
        corrected_labels = list(labels)

        idx = 0
        while idx < len(corrected_labels):
            label = corrected_labels[idx]

            if label == 'normal':
                idx += 1
                continue

            elif label == 'extra':  # Interval is too short
                print("extra is founded")
                if idx + 1 < len(rr_corrected):
                    rr_corrected[idx] += rr_corrected[idx + 1]  # The short interval is added to its following interval
                    del rr_corrected[idx + 1]
                    del corrected_labels[idx + 1]
                idx += 1

            elif label == 'missed':  # Interval is too long
                half = rr_corrected[idx] / 2  # The long interval is being split into two
                rr_corrected[idx] = half
                rr_corrected.insert(idx + 1, half)
                corrected_labels.insert(idx + 1, 'missed')
                idx += 2

            elif label in ['ectopic', 'long/short']:
                prev_idx = idx - 1
                next_idx = idx + 1
                while prev_idx >= 0 and corrected_labels[prev_idx] != "normal":
                    prev_idx -= 1
                while next_idx < len(rr_corrected) and corrected_labels[next_idx] != "normal":
                    next_idx += 1
                if 0 <= prev_idx < len(rr_corrected) and next_idx < len(rr_corrected):
                    rr_corrected[idx] = (rr_corrected[prev_idx] + rr_corrected[next_idx]) / 2
                elif 0 <= prev_idx < len(rr_corrected):
                    rr_corrected[idx] = rr_corrected[prev_idx]
                elif next_idx < len(rr_corrected):
                    rr_corrected[idx] = rr_corrected[next_idx]
                else:
                    rr_corrected[idx] = np.nan
                idx += 1
            else:
                idx += 1

        return rr_corrected, corrected_labels

    def correct_and_classify_rr(self, rr_intervals):
        """
        Detect, classify, and correct artefacts from an RR interval series.

        Args:
            rr_intervals (list): Original RR interval series, of one participant.

        Returns:
            tuple:
                corrected_rr (list): Corrected RR intervals,
                corrected_labels (list): Beat classification labels.
                correction_percent (float): % of beats corrected,
        """
        dRRs = self.calculate_dRRs(rr_intervals)
        mRRs = self.calculate_mRRs(rr_intervals)

        Th1 = self.calculate_threshold(dRRs)
        Th2 = self.calculate_threshold(mRRs)

        dRR_norm = self.normalize_series(dRRs, Th1)
        mRR_norm = self.normalize_series(mRRs, Th2)

        labels = self.classify_beats(dRR_norm, mRR_norm, rr_intervals, Th2)
        corrected_rr, corrected_labels = self.correct_rr_intervals(rr_intervals, labels)

        abnormal_indices = [i for i, label in enumerate(labels) if label != "normal"]
        correction_percent = (len(abnormal_indices) / len(rr_intervals)) * 100

        return corrected_rr, corrected_labels, correction_percent

    def correct_all_data(self, data):
        """
        Apply correction to all participants and phases.

        Args:
            data (dict): Nested dictionary of original data.

        Returns:
            dict: Nested dictionary with corrected data and labels.
        """
        corrected_data = {}
        for participant_id, phases in data.items():
            print(f"analyzing {participant_id}...")
            corrected_data[participant_id] = {}
            for phase_name, rr_intervals in phases.items():
                corrected_rr, labels, correction_percent = self.correct_and_classify_rr(rr_intervals)
                corrected_data[participant_id][phase_name] = {
                    "corrected_rr": corrected_rr,
                    "correction_percent": correction_percent,
                    "labels": labels
                }
        return corrected_data

    def save_corrected_data(self, corrected_data, save_dir="corrected_data"):
        """
        Save corrected RRIs and beat labels into Excel files, one per participant.

        Args:
            corrected_data (dict): Nested corrected data.
            save_dir (str): Directory where Excel files will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)

        for participant_id, phases in corrected_data.items():
            save_path = os.path.join(save_dir, f"{participant_id}.xlsx")
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                for phase_name, content in phases.items():
                    df = pd.DataFrame({
                        "Corrected_RRI": content["corrected_rr"],
                        "Label": content["labels"]
                    })

                    # Add bottom row with correction percentage
                    df.loc[len(df)] = [f"Correction Percentage:", f"{content['correction_percent']:.2f}%"]

                    df.to_excel(writer, sheet_name=phase_name, index=False)

    def plot_correction(self, original_rr, corrected_rr, title="RR Interval Correction"):
        """
        Plot original vs corrected RR interval series for comparison.

        Args:
            original_rr (list): Original RR interval series.
            corrected_rr (list): Corrected RR interval series.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(original_rr, label="Original", color='red', alpha=0.7)
        plt.plot(corrected_rr, label="Corrected", color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel("Beat number")
        plt.ylabel("RR Interval (ms)")
        plt.legend()
        plt.grid()
        plt.show()


def save_correction_summary(corrected_data, save_path):
    summary_rows = []

    for participant_id, phases in corrected_data.items():
        row = {"ID": participant_id}
        for phase in ["baseline", "task", "recovery"]:
            if phase in phases:
                row[phase] = round(phases[phase]["correction_percent"], 2)
            else:
                row[phase] = None  # or 0.0 if you prefer
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[["ID", "baseline", "task", "recovery"]]  # Ensure column order
    file_name = "correction_summary.xlsx"
    save_path = os.path.join(save_path, file_name)
    summary_df.to_excel(save_path, index=False)
    print(f"Correction summary saved to {save_path}")
