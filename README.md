# SafePath
SIADS 699 Capstone Project - Authored by Kevin Leander and Tanzim Chowdhury

SafePath is a data science project that models street-level traffic crash risk in urban environments. Rather than only identifying historical crash hotspots, the goal is to estimate **situational crash risk** by combining infrastructure characteristics, traffic exposure, and environmental conditions. The project uses spatial and temporal data to predict where and when crashes are more likely to occur across city street segments.

## Objective

The main objective is to build predictive models that estimate crash risk for individual street segments over time. By integrating multiple datasets, the project aims to identify which infrastructure, traffic, and environmental factors are most strongly associated with traffic collisions. The final outputs will include predictive models, spatial visualizations of risk patterns, and analysis of key drivers of urban roadway safety.

## Data Sources

The project integrates several publicly available datasets:

- **NYC Motor Vehicle Collisions (NYC Open Data)** – crash records including time, location, and injury severity.
- **OpenStreetMap Road Network (via OSMnx)** – street geometry, road classification, and intersection structure.
- **NYC DOT Traffic Volume Counts** – traffic exposure indicators such as Average Annual Daily Traffic (AADT).
- **Weather Data (NOAA / Open-Meteo API)** – precipitation, temperature, wind, and other environmental variables.

## Repository Structure

SafePath/  
│  
├── src/  
│   ├── data/          # Data ingestion scripts  
│   ├── processing/    # Data cleaning and spatial processing  
│   ├── features/      # Feature engineering  
│   └── modeling/      # Predictive modeling  
│  
├── notebooks/         # Exploratory analysis  
│  
├── data/  
│   ├── raw/           # Raw datasets (not tracked in Git)  
│   ├── interim/       # Intermediate geospatial datasets  
│   └── processed/     # Final modeling datasets  
│  
├── requirements.txt  
└── README.md  

## Current Data Pipeline

1. **Crash Data Ingestion** – downloads NYC collision data (2018–2024).  
2. **Crash Data Cleaning** – converts crash records to geospatial points and removes invalid coordinates.  
3. **Street Network Extraction** – downloads the NYC drivable street network using OSMnx.  
4. **Spatial Join** – assigns each crash to its nearest street segment.

Future pipeline stages will include crash aggregation by segment and time, integration of weather and traffic data, feature engineering, and predictive modeling.

## Dataset Availability

Due to file size limitations, datasets are **not stored in this repository**. Many intermediate geospatial datasets exceed several hundred megabytes, and GitHub limits files to **100 MB**.

Instead, this repository contains scripts that automatically download and generate the required datasets through the data pipeline.

To reproduce the datasets locally, run the scripts in the following order:

python src/data/fetch_crash_data.py  
python src/processing/clean_crash_data.py  
python src/data/fetch_osm_network.py  
python src/processing/map_crashes_to_segments.py  

## Requirements

Install dependencies with:

pip install -r requirements.txt  

Key libraries used in this project include `pandas`, `geopandas`, `osmnx`, `shapely`, `scikit-learn`, and `requests`.

## Status

This project is currently under development as part of the **SIADS 699 Capstone Project** in the University of Michigan Master of Applied Data Science program.