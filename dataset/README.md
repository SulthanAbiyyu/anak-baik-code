# Dataset Info

## Dataset Stages

### 1. Landing

- **Description**: This is the initial stage where data is first acquired and brought into the system. The landing phase involves downloading the raw data from external sources. This step ensures that all incoming data is stored in a centralized location before any processing takes place.

- **Key Actions**:
  - Downloading data from various sources.
  - Storing the data in a raw format.

### 2. Raw

- **Description**: In this stage, the raw data is processed to select relevant columns and filter out any inconsistencies or unnecessary information. The goal is to standardize the data and make it more manageable for subsequent processing stages.

- **Key Actions**:
  - Selecting relevant columns based on the needs of the project.
  - Filtering data to remove outliers or irrelevant entries.
  - Ensuring consistency in data formats and values.

### 3. Transformed

- **Description**: The transformed stage involves reshaping the data into the desired format or schema. This phase include translating the data, constructing rejection responses, and applying other transformations to make the data suitable for analysis or integration into the systems.

- **Key Actions**:
  - Converting data into the required schema or format.
  - Translating text data if necessary.
  - Creating or constructing rejection responses for data that does not meet the criteria.

### 4. Curated

- **Description**: In the curation stage, the data is organized, split into relevant subsets, and unified to ensure consistency.

- **Key Actions**:
  - Splitting data into training, validation, and test sets if applicable.
  - Unifying datasets to ensure consistent formatting and schema across different subsets.

## Dataset Schema

1. **instruction (String)**: Describes the instructional content or task.
2. **input (optional) (String)**: Optional input required for the instruction.
3. **output (String)**: Expected output of the instruction.
4. **split (String)**: Indicates whether the data belongs to the training or test set (`train`, `test`).
5. **type (String)**: Categorization of the instruction type (`harmless`, `toxic`, `umum`, etc.).
6. **topic (String)**: Specifies the subject or theme of the instruction.

## Usage

```python
python3 etl.py
```

## Logs
### Performance Logs
```plaintext
Elapsed time for save_dataset_to_csv is : 4.99 seconds # HarmfulQA
Elapsed time for save_dataset_to_csv is : 10.43 seconds # IndoToxic2024
Elapsed time for save_dataset_to_csv is : 4.23 seconds # Alpaca
Elapsed time for load_and_save_datasets is : 43.68 seconds
Elapsed time for process_hqa_data is : 0.68 seconds
Elapsed time for process_alpaca_data is : 2.01 seconds
Elapsed time for process_toxic_data is : 0.43 seconds
Elapsed time for process_hqa_data is : 0.03 seconds
Elapsed time for process_toxic_data is : 0.03 seconds
Elapsed time for process_alpaca_data is : 0.69 seconds
Elapsed time for split_data is : 0.00 seconds
Elapsed time for split_data is : 0.00 seconds
Elapsed time for split_data is : 0.00 seconds
Elapsed time for unify is : 0.19 seconds
```

### Changelogs
1. Add `topic` in the schema 
2. Change the data lifecycle into LRTC (Landing, Raw, Tranform, Curate)

## TODO:

1. Translate and create self-evalation logic in `curate.py`

