import os

DATASET_DIR = "dataset"  # root folder containing A/, B/, ..., nothing/
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def count_images_per_class(root):
    counts = {}
    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        count = 0
        for fname in os.listdir(cls_path):
            if os.path.splitext(fname)[1].lower() in VALID_EXTENSIONS:
                count += 1
        counts[cls] = count
    return counts

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print(f"⚠️ Dataset folder '{DATASET_DIR}' not found.")
    else:
        counts = count_images_per_class(DATASET_DIR)
        total = sum(counts.values())
        print("Image counts per class:")
        for cls, n in counts.items():
            print(f"  {cls:10s} : {n:5d}")
        print(f"Total images: {total}")
