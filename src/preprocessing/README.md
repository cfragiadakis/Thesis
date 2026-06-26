# DIPSER Preprocessing

This directory contains the preprocessing code that converts the raw DIPSER
release into the tabular parquet dataset used by the experiment notebooks:

```text
Data/dipser_dataset.parquet
```

DIPSER is larger than 660 GB and stores image frames, smartwatch sensor streams,
labels, and metadata in separate files. The preprocessing is therefore split
into five tasks:

1. Extract one initial CSV per subject from the raw folder structure.
2. Align smartwatch sensor readings to image frames.
3. Extract demographic metadata for each frame.
4. Validate frame metadata and mark unusable frames.
5. Build the final cleaned dataset used by the models.

Run all commands from the project root.

## Raw Data Layout

Download DIPSER from the official dataset page:

https://www.scidb.cn/en/detail?dataSetId=7856c716c0cc4589a23ee4a23d8a0893#p2

The extracted data should follow this structure:

```text
Data/DIPSER/
  group01/
    experiment01/
      subject_01/
        images/
        labels/
        metadata/ or metadata.tar
        watch_sensors/
```

## Run the Full Pipeline

The complete pipeline can be run with:

```bash
python -m src.preprocessing.preprocessing_pipeline \
  --raw-root Data/DIPSER \
  --output-dir Data \
  --num-workers 4
```

If preprocessing was interrupted, reuse existing subject CSVs and aggregated
subject parquet files:

```bash
python -m src.preprocessing.preprocessing_pipeline \
  --raw-root Data/DIPSER \
  --output-dir Data \
  --num-workers 4 \
  --skip-existing
```

To process only part of the dataset, add one or more filters:

```bash
--group group01
--experiment experiment01
--subject subject_01
```

## Step 1: Extract Subject CSVs

The first step scans the raw DIPSER folders and creates one CSV file per subject.
These CSVs combine image rows, label references, metadata paths, and raw sensor
rows into a format that later steps can process more efficiently.

Input:

```text
Data/DIPSER/group*/experiment*/subject*/
```

Output:

```text
Data/extracted_data_extended/group*/experiment*/subject*.csv
```

Command:

```bash
python -m src.preprocessing.initial_processor \
  --base_path Data/DIPSER \
  --output_path Data/extracted_data_extended
```

Resume without recomputing existing CSV files:

```bash
python -m src.preprocessing.initial_processor \
  --base_path Data/DIPSER \
  --output_path Data/extracted_data_extended \
  --skip-existing
```

## Step 2: Aggregate Sensor Data Around Frames

The second step aligns smartwatch sensor readings to the image timeline. For
each image frame, sensor values within a `+/-500 ms` window are summarized with
mean and standard deviation features. This turns irregular sensor streams into
frame-level model features.

Input:

```text
Data/extracted_data_extended/group*/experiment*/subject*.csv
```

Outputs:

```text
Data/extracted_data_aggregated/group*/experiment*/subject*.parquet
Data/initial_data.parquet
```

Command:

```bash
python -m src.preprocessing.aggregate_sensor_data \
  --input-dir Data/extracted_data_extended \
  --output-dir Data/extracted_data_aggregated \
  --initial-data Data/initial_data.parquet \
  --num-workers 4
```

Resume without recomputing existing subject parquet files:

```bash
python -m src.preprocessing.aggregate_sensor_data \
  --input-dir Data/extracted_data_extended \
  --output-dir Data/extracted_data_aggregated \
  --initial-data Data/initial_data.parquet \
  --num-workers 4 \
  --skip-existing
```

## Step 3: Extract Demographic Metadata

The third step reads the per-frame metadata and adds demographic columns:
estimated age, gender, dominant race, and race probabilities. It supports both
extracted JSON metadata files and `metadata.tar` archives.

Input:

```text
Data/initial_data.parquet
```

Output:

```text
Data/initial_metadata.parquet
```

Command:

```bash
python -m src.preprocessing.extract_metadata \
  --input Data/initial_data.parquet \
  --output Data/initial_metadata.parquet
```

## Step 4: Validate Frame Metadata

The fourth step checks whether each frame has usable face metadata. Frames are
marked invalid when face metadata is missing or the face bounding box is missing or malformed.

Input:

```text
Data/initial_metadata.parquet
```

Output:

```text
Data/dipser_data_valid.parquet
```

Command:

```bash
python -m src.preprocessing.validate_frame_metadata \
  --input Data/initial_metadata.parquet \
  --output Data/dipser_data_valid.parquet
```

## Step 5: Build the Final Dataset

The final step performs the cleaning used before training and evaluation:

- removes invalid frames
- converts frame-level demographic estimates into subject-level labels
- creates subject-level `age_group` bins
- removes sessions with too much missing sensor data
- creates the final `attention` target from non-self labeler annotations
- adds `subject_id`, `subject_experiment_id`, and `time_sec`

Input:

```text
Data/dipser_data_valid.parquet
```

Output:

```text
Data/dipser_dataset.parquet
```

Command:

```bash
python -m src.preprocessing.dipser_dataset \
  --input Data/dipser_data_valid.parquet \
  --output Data/dipser_dataset.parquet
```
