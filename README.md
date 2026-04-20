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