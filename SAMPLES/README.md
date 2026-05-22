This repository contains sample Apache Parquet files of the Medical Hindi Image Dataset. The dataset includes sample medical images stored in compressed binary format for efficient machine learning workflows, computer vision research, dataset evaluation, and AI model development.

Convert Parquet Back to Images

# Requirements

Install dependencies:

```bash
pip install pandas pillow pyarrow pydicom numpy
```

Use the following Python script to extract data from the Parquet file.
```python
import os
import io
import pandas as pd

from PIL import Image

PARQUET_FILE = r"C:\Users\3\Downloads\PA01.parquet"

OUTPUT_FOLDER = r"C:\Users\3\Downloads\PA01Restored_Dataset"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_parquet(PARQUET_FILE)

for index, row in df.iterrows():

    patient_id = str(row["patient_id"])

    findings = str(row["findings"])

    body_part = str(row["body_part"])

    sex = str(row["sex"])

    study_uid = str(row["study_uid"])

    # -----------------------------------
    # Create patient folder
    # -----------------------------------

    patient_folder = os.path.join(
        OUTPUT_FOLDER,
        patient_id
    )

    os.makedirs(
        patient_folder,
        exist_ok=True
    )

    # -----------------------------------
    # Extract image
    # -----------------------------------

    image = row["image"]

    image_path = os.path.join(
        patient_folder,
        f"{patient_id}.png"
    )

    try:

        # HuggingFace image dictionary
        if isinstance(image, dict):

            # Case 1: image bytes
            if "bytes" in image:

                img = Image.open(
                    io.BytesIO(image["bytes"])
                )

                img.save(image_path)

            # Case 2: image path
            elif "path" in image:

                img = Image.open(image["path"])

                img.save(image_path)

        # PIL image directly
        elif isinstance(image, Image.Image):

            image.save(image_path)

        print("Saved Image:", image_path)

    except Exception as e:

        print("Image Error:", patient_id)
        print(e)

    # -----------------------------------
    # Save metadata
    # -----------------------------------

    metadata_path = os.path.join(
        patient_folder,
        "metadata.txt"
    )

    with open(metadata_path, "w", encoding="utf-8") as f:

        f.write(f"Patient ID: {patient_id}\n")
        f.write(f"Body Part: {body_part}\n")
        f.write(f"Sex: {sex}\n")
        f.write(f"Study UID: {study_uid}\n\n")
        f.write("Findings:\n")
        f.write(findings)

    print("Saved Metadata:", metadata_path)

print("\nCompleted")
```


# Considerations

These Parquet files contain a sample of the complete dataset corpus and are provided for preview, evaluation, testing, and research purposes. The files are optimized in the Apache Parquet format for efficient storage and fast loading.

Please note that the uploaded files do not represent the full dataset collection. They include only a limited portion of the overall corpus intended to demonstrate the dataset structure, schema, and content quality.

For access to the full dataset, custom data delivery, commercial usage, or enterprise licensing options, please visit [InfoBay AI](https://infobay.ai/) or contact us directly for further information.

    -Ph: (91) 8303174762
    -Email: datareq@infobay.ai
