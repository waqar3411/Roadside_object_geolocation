python scripts/run_amsterdam_inference.py \
    --image-dir /path/to/Amsterdam/New_testing_unique_images_captured_at \
    --metadata-csv /path/to/Amsterdam/New_testing_unique_images_captured_at.csv \
    --model-path /path/to/trained_model.pkl \
    --output-dir /path/to/Amsterdam/New_testing_out


The script creates confidence-based folders for positive predictions:
5_to_6/
6_to_8/
8_to_9/
greater_9/

It also generates CSV files containing prediction confidence scores and metadata. The main metadata file is:
Ams_All.csv

This file contains the detected crop name, camera location, object/view bearing, field-of-view information, and capture date.
It serves as the intermediate detection metadata file that can be passed to the MRF geolocation module to estimate final object positions.
