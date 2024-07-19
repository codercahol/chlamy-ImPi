#!/bin/bash

# Activate conda env
conda activate chlamy

# Download latest data
cd ../data
./download_latest_data_simple.sh

# Generate segmentations
cd ../chlamy_impi/well_segmentation_preprocessing
python main.py

# Create database
cd ../database_creation
python main.py

# Rename database using today's date
cd ../../output/database_creation
mv database.parquet database_$(date +%Y-%m-%d).parquet