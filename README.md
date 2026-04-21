# SafePath
SIADS 699 Capstone Project – Authored by Kevin Leander and Tanzim Chowdhury

SafePath is a data science project that models street-level traffic crash risk in urban environments. Rather than only identifying historical crash hotspots, the project estimates **situational crash risk** by combining infrastructure characteristics, traffic exposure, and environmental conditions. The goal is to move beyond reactive safety analysis and toward **proactive identification of high-risk roadway conditions**.

---

## Objective

The primary objective of SafePath is to build predictive models that estimate crash risk for individual street segments over time. By integrating multiple data sources, the project identifies which roadway, environmental, and temporal factors are most strongly associated with collisions.

The final outputs include:
- Predictive risk scores for street segments
- Spatial visualizations of crash patterns and model outputs
- Infrastructure-based insights into roadway safety
- An interactive dashboard for exploration and analysis

---

## Data Sources

The project integrates several publicly available datasets:

- **NYC Motor Vehicle Collisions (NYC Open Data)**  
  Crash records including timestamp, location, and severity.

- **OpenStreetMap Road Network (via OSMnx)**  
  Street geometry, road classification, and intersection topology.

- **NYC DOT Traffic Volume Counts**  
  Traffic exposure indicators such as Average Annual Daily Traffic (AADT).

- **Weather Data (NOAA / Open-Meteo API)**  
  Environmental variables including precipitation, temperature, and wind.

---

## Data Access

All datasets used in this project are publicly available:

- NYC Motor Vehicle Collisions: NYC Open Data  
- OpenStreetMap road network via OSMnx  
- NYC DOT traffic counts (if used)  
- Weather data via the Open-Meteo API  

Due to file size limitations, raw and processed datasets are not stored in this repository.

To reproduce results:
1. Run data ingestion scripts in `src/data/`
2. Run processing and feature scripts in `src/processing/` and `src/features/`

All data usage complies with the respective licenses of each source.

---

## Methodology Overview

SafePath constructs a **spatiotemporal modeling pipeline**:

1. **Data Integration**
   - Map crash events to street segments
   - Align temporal features at a daily level
   - Merge infrastructure, traffic, and weather features

2. **Feature Engineering**
   - Rolling crash history (lag features)
   - Infrastructure attributes (lanes, speed, curvature, intersections)
   - Environmental conditions
   - Custom metrics such as a **visibility risk score**

3. **Modeling**
   - Baseline models (logistic regression with and without lag features)
   - Gradient boosting models (LightGBM) for final predictions
   - Evaluation using ROC AUC, Average Precision, and classification metrics

4. **Unsupervised Learning**
   - **Gaussian Mixture Models (GMM)** to identify roadway archetypes
   - **Hidden Markov Models (HMM)** to capture temporal crash regimes

---

## Dashboard Application

SafePath includes an interactive dashboard built with **Next.js and React Leaflet**.

### Key Features

- **Crash Map**
  - Historical crash intensity visualization
  - Predicted crash risk from the trained model
  - Infrastructure overlays (lanes, speed, traffic signals, visibility)
  - Segment-level filtering and interaction

- **Insights Page**
  - Model feature importance
  - Cluster-based roadway archetypes
  - Temporal crash regime summaries
  - Key takeaways and limitations

Due to file size constraints, large geospatial datasets are hosted externally and accessed via environment variables rather than being stored directly in the repository.

---

## Repository Structure

SafePath/
│
├── src/
│   ├── data/                    # Data ingestion scripts
│   ├── processing/              # Data cleaning and spatial processing
│   ├── features/                # Feature engineering
│   └── modeling/                # Predictive modeling
│
├── notebooks/                   # Exploratory and results analysis
│
├── data/
│   ├── raw/                     # Raw datasets (not tracked in Git)
│   ├── interim/                 # Intermediate geospatial datasets
│   └── processed/               # Final modeling datasets
│
├── models/                      # Trained model artifacts and metrics
├── safepath-dashboard/          # Frontend dashboard (Next.js app)
├── requirements.txt
└── README.md

---

## Data Pipeline

The pipeline constructs a segment-level dataset with temporal features:

1. Fetch crash, weather, and network data
2. Clean and geocode crash records
3. Map crashes to street segments
4. Aggregate crashes by segment and time
5. Merge infrastructure and environmental features
6. Generate model-ready datasets
7. Train predictive models and export outputs

Example execution:

python src/data/fetch_crash_data.py  
python src/processing/clean_crash_data.py  
python src/data/fetch_osm_network.py  
python src/processing/map_crashes_to_segments.py  

---

## How to Run

Install dependencies:

pip install -r requirements.txt

---

### Step 1: Fetch raw data
These scripts download all primary datasets used in the project, including crash data, the street network, and weather data.

python src/data/fetch_crash_data.py  
python src/data/fetch_osm_network.py  
python src/data/fetch_weather_data.py  

---

### Step 2: Clean and prepare crash data
python src/processing/clean_crash_data.py

---

### Step 3: Build the street segment base
python src/processing/build_street_segments.py  
python src/processing/map_crashes_to_segments.py  
python src/processing/aggregate_segment_crashes.py  

---

### Step 4: Build the modeling dataset
python src/processing/build_segment_day_panel_lightgbm.py  
python src/processing/merge_datasets.py  
python src/processing/merge_weather_features.py  
python src/processing/sample_negative_examples.py  

---

### Step 5: Feature engineering
python src/features/build_segment_features.py  
python src/features/build_temporal_crash_features.py  
python src/features/build_infrastructure_features.py  
python src/features/merge_infrastructure_into_full_panel.py  
python src/features/build_segment_profiles.py  
python src/features/build_segment_cluster_dataset.py  
python src/features/build_hmm_daily_dataset.py  
python src/features/finalize_model_features.py  

---

### Step 6: Train models and generate predictions
python src/modeling/baseline_model.py  
python src/modeling/baseline_no_lag_model.py  
python src/modeling/lightgbm_full_panel_model.py  
python src/modeling/generate_full_panel_predictions.py  
python src/modeling/train_gmm_segment_clusters.py  
python src/modeling/train_hmm_daily_regimes.py  

---

### Step 7: Export dashboard-ready outputs
python src/processing/export_dashboard_segment_map.py  

---

## Notes

Large datasets are not stored in this repository due to file size limits. All data can be reproduced by running the pipeline above.

The dashboard reads from a hosted GeoJSON file. Set the following environment variable before running the dashboard:

NEXT_PUBLIC_SEGMENT_MAP_URL="https://pub-de0b628f20714f42a9f82a56b1b3ce59.r2.dev/map.geojson"

---

## Dataset Availability

Due to GitHub’s **100 MB file limit**, large datasets are not included in this repository.

- Raw datasets are retrieved via scripts  
- Intermediate datasets are generated locally  
- Final dashboard data is hosted externally  

This ensures reproducibility while keeping the repository lightweight.

---

## Requirements

Install dependencies with:

pip install -r requirements.txt  

Key libraries:
- pandas  
- numpy  
- geopandas  
- osmnx  
- shapely  
- scikit-learn  
- lightgbm  
- requests  

---

## Code Attribution

The interactive map implementation using React Leaflet, along with portions of the dashboard UX/UI design (including map controls, filtering interactions, and layout structure), were developed with the assistance of AI tools (ChatGPT).

All core data processing, feature engineering, modeling, and analysis logic were implemented by the project authors.

---

## Project Status

This project was developed as part of the **SIADS 699 Capstone Project** in the University of Michigan Master of Applied Data Science program.

The current version includes:
- An end-to-end data pipeline  
- Trained predictive models  
- Unsupervised analysis (GMM and HMM)  
- A deployed interactive dashboard  

Future improvements may include:
- Incorporating real-time data sources  
- Enhancing model calibration and interpretability  
- Expanding to additional cities  