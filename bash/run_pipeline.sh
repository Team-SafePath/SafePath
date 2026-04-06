#!/bin/bash

set -e

# Move to project root (one level up from bash/)
cd "$(dirname "$0")/.."

echo "Running SafePath data pipeline..."

python src/data/fetch_crash_data.py
python src/processing/clean_crash_data.py
python src/data/fetch_osm_network.py
python src/processing/map_crashes_to_segments.py
python src/processing/aggregate_segment_crashes.py
python src/processing/sample_negative_examples.py
python src/data/fetch_weather_data.py
python src/processing/merge_weather_features.py
python src/features/build_segment_features.py
python src/processing/merge_segment_features.py
python src/features/build_temporal_crash_features.py
python src/features/finalize_model_features.py

echo "Pipeline complete!"