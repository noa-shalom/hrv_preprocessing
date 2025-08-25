# HRV Preprocessing Pipeline

A Python pipeline for preprocessing Polar RR interval (RRI) data, correcting artifacts, and computing heart rate variability (HRV) metrics.  
The artifact correction is based on **Lipponen & Tarvainen (2019)**, and the pipeline outputs cleaned data, summary metrics (RMSSD), and visualizations.

---

## Features
- Load raw Polar RRI data (organized by participant and phase).
- Detect, classify and correct artifacts in RR intervals.
- Save corrected RRIs and correction summaries.
- Compute RMSSD (per baseline, task, and recovery).
- Generate Excel summaries and boxplots.

---

## Installation
Clone the repository and install the requirements:

```bash
git clone https://github.com/noa-shalom/hrv_preprocessing.git
cd hrv_preprocessing
pip install -r requirements.txt
```

## Usage
Run the main script locally:
```bash
python src/main.py
```

**Input data structure:**
Place your raw data under a raw_data/ folder inside the project.
Each participant should have a subfolder with three text files:
- baseline
- task
- recovery

**Outputs:**
- Corrected RRIs per participant (corrected_data/)
- Summary of corrections percentage (results/correction_summary.xlsx)
- RMSSD calculated data (results/rmssd_summary.xlsx)
- Boxplots of RMSSD (results/rmssd.png)
- Cleaned results without outliers (results/rmssd_no_outliers.*)

---

## Reference
Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time series artefact correction. Physiological Measurement, 40(6).

---
# Author
Developed by Noa Shalom
