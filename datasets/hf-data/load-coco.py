# This script temporary test loader.py

from datasets import load_dataset

# The path to your local script
script_path = "coco17.py"

print(f"Loading dataset using local script: {script_path}")

# Load the validation split to test. This will trigger the download.
# Use streaming=True for a quick check without downloading everything.
try:
    # Set trust_remote_code=True to allow execution of your local script
    coco_val_dataset = load_dataset(script_path, trust_remote_code=True)

    print("\nDataset loaded successfully!")
    print(f"Dataset features: {coco_val_dataset.features}")

    # Print the first example
    print("\nFirst example:")
    first_example = next(iter(coco_val_dataset))
    print(first_example)

except Exception as e:
    print(f"An error occurred: {e}")
