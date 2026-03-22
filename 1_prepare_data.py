import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================

DATASET_DIR = "The IQ-OTHNCCD lung cancer dataset"
OUTPUT_DIR = "data"
IMG_SIZE = 256

# dataset folder names → clean class names
CLASS_MAP = {
    "Bengin cases": "Benign",
    "Malignant cases": "Malignant",
    "Normal cases": "Normal"
}

# =========================
# CREATE OUTPUT FOLDERS
# =========================

for split in ["train", "val", "test"]:
    for cls in CLASS_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

print("Output folders created")

# =========================
# IMAGE PROCESSING
# =========================

def process_and_save(src, dst):

    try:

        img = Image.open(src)

        # convert grayscale to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((IMG_SIZE, IMG_SIZE))

        img.save(dst, quality=95)

    except:
        print("Skipping corrupted image:", src)


# =========================
# DATASET PREPARATION
# =========================

for folder in CLASS_MAP:

    class_dir = os.path.join(DATASET_DIR, folder)

    if not os.path.exists(class_dir):
        print("Folder not found:", class_dir)
        continue

    print("Processing:", folder)

    images = os.listdir(class_dir)

    train, temp = train_test_split(images, test_size=0.3, random_state=42)

    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    splits = {
        "train": train,
        "val": val,
        "test": test
    }

    for split, files in splits.items():

        for img in files:

            src = os.path.join(class_dir, img)

            dst = os.path.join(
                OUTPUT_DIR,
                split,
                CLASS_MAP[folder],
                img
            )

            process_and_save(src, dst)

print("\nDataset preparation complete")

print("\nFinal dataset structure:")
print("data/")
print(" ├── train/")
print(" ├── val/")
print(" └── test/")