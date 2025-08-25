from rr_artifact_correction import RRArtifactCorrector, save_correction_summary
from hrv_metrics import compute_all_rmssd, save_results, create_boxplots, remove_outliers
import os

# A script for cleaning Polar RRI data and calculating RMSSD.
# Written by Noa Shalom, based on the article by Lipponen & Tarvainen (2019).

corrector = RRArtifactCorrector()

# Step 1: Load the raw data
raw_data_path = r"C:\Users\noaaa\PycharmProjects\hrv_artifacts_correction\raw_data"
data = corrector.load_data(raw_data_path)

# Step 2: Correct the data
# Note: see all steps in the rr_artifact_correction module
corrected_data = corrector.correct_all_data(data)

# Step 3: Save corrected data
corrected_data_path = r"C:\Users\noaaa\PycharmProjects\hrv_artifacts_correction\corrected_data"
results_path = r"C:\Users\noaaa\PycharmProjects\hrv_artifacts_correction\results"
corrector.save_corrected_data(corrected_data, save_dir=corrected_data_path)
save_correction_summary(corrected_data, save_path=results_path)

# # Step 4 (Optional): Plot an example
# participant = 'rn24031'
# phase = 'task'
# original_rr = data[participant][phase]
# corrected_rr = corrected_data[participant][phase]['corrected_rr']
# corrector.plot_correction(original_rr, corrected_rr, title=f"{participant} - {phase}")

# Step 5: Compute RMSSD and store the results
results = compute_all_rmssd(corrected_data)
rmssd_df = save_results(results, results_path, "rmssd_summary.xlsx")

# Step 6: Boxplots of the results
create_boxplots(rmssd_df, results_path, "rmssd.png")

# Step 7: Remove outliers and re-save results
cleaned_rmssd_df = remove_outliers(rmssd_df)
path = os.path.join(results_path, "rmssd_no_outliers.xlsx")
cleaned_rmssd_df.to_excel(path, index=True)
create_boxplots(cleaned_rmssd_df, results_path, "rmssd_no_outliers.png")
