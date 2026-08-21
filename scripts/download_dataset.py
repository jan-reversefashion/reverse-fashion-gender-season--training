"""Download images from S3 into an ImageFolder directory structure.

Usage:
    uv run scripts/download_dataset.py --target-field category_lvl0 --output data/category_lvl0
    uv run scripts/download_dataset.py --target-field demography --no-balance --output data/demography
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("FIFTYONE_API_URI", "https://reverse-fashion-api.fiftyone.ai")
if "FIFTYONE_API_KEY" not in os.environ:
    raise EnvironmentError("Set FIFTYONE_API_KEY in .env or as an environment variable")

import fiftyone as fo


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset images from S3 into ImageFolder structure")
    parser.add_argument("--target-field", default="category_lvl0", help="FiftyOne field to classify")
    parser.add_argument("--category-lvl1-filter", default="Clothing", help="Filter category_lvl1 value")
    parser.add_argument("--group-slice", default="0", help="Group slice to select")
    parser.add_argument("--output", default="/home/sagemaker-user/user-default-efs/data/sellpy2", help="Output directory (ImageFolder structure)")
    parser.add_argument("--balance", action="store_true", default=True, help="Balance classes by subsampling")
    parser.add_argument("--no-balance", dest="balance", action="store_false")
    parser.add_argument("--workers", type=int, default=16, help="Parallel download threads")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def download_image(s3_client, s3_path, dest_path):
    """Download a single image from S3."""
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(dest_path))


def main():
    args = parse_args()
    random.seed(args.seed)

    TAG = "gender_balance"

    print("Loading dataset 'sellpy2' from FiftyOne...")
    dataset = fo.load_dataset("sellpy2")

    view = dataset.match(fo.ViewField("category_lvl1") == args.category_lvl1_filter)
    view = view.match(fo.ViewField(args.target_field).exists())
    view = view.select_group_slices(args.group_slice)
    print(f"Samples with '{args.target_field}' populated: {len(view)}")

    # Check for previously tagged samples
    tagged_view = view.match_tags(TAG)
    if len(tagged_view) > 0:
        print(f"Found {len(tagged_view)} samples already tagged '{TAG}' — using those")
        filepaths = tagged_view.values("filepath")
        target_values = tagged_view.values(args.target_field)
    else:
        filepaths = view.values("filepath")
        target_values = view.values(args.target_field)

    sample_ids = tagged_view.values("id") if len(tagged_view) > 0 else view.values("id")

    unique_labels = sorted(set(l for l in target_values if l is not None))
    class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Classes ({len(unique_labels)}): {unique_labels}")
    print(f"Distribution: {Counter(target_values).most_common(10)}")

    all_samples = [
        (fp, lbl, sid)
        for fp, lbl, sid in zip(filepaths, target_values, sample_ids)
        if lbl in class_to_idx
    ]
    print(f"Total usable samples: {len(all_samples)}")

    if args.balance and len(tagged_view) == 0:
        samples_by_class = defaultdict(list)
        for item in all_samples:
            samples_by_class[item[1]].append(item)
        min_count = min(len(v) for v in samples_by_class.values())
        all_samples = []
        for cls in sorted(samples_by_class):
            all_samples.extend(random.sample(samples_by_class[cls], min_count))
        print(f"After balancing: {len(all_samples)} samples ({min_count} per class)")

        # Tag selected samples in FiftyOne so they are reused next time
        selected_ids = [sid for _, _, sid in all_samples]
        selected_view = dataset.select(selected_ids)
        selected_view.tag_samples(TAG)
        print(f"Tagged {len(selected_ids)} samples with '{TAG}'")

    sample_data = [(fp, lbl) for fp, lbl, _ in all_samples]

    # Download images into ImageFolder structure: output/<label>/filename.jpg
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(sample_data)} images to {output_dir}/ ...")

    def _download_one(item):
        filepath, label = item
        filename = Path(filepath).name
        dest = output_dir / label / filename
        if dest.exists():
            return True
        s3 = boto3.client("s3")
        download_image(s3, filepath, dest)
        return True

    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_download_one, item): item for item in sample_data}
        for future in as_completed(futures):
            try:
                future.result()
                downloaded += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  WARN: failed to download {futures[future][0]}: {e}")
            if (downloaded + failed) % 500 == 0:
                print(f"  Progress: {downloaded + failed}/{len(sample_data)}")

    print(f"Done: {downloaded} downloaded, {failed} failed")

    # Save class mapping
    meta_path = output_dir / "classes.json"
    with open(meta_path, "w") as f:
        json.dump({"target_field": args.target_field, "class_to_idx": class_to_idx}, f, indent=2)

    print(f"Dataset saved to {output_dir}/")
    print(f"  Structure: {output_dir}/<class_name>/<image>.jpg")
    print(f"  Class mapping: {meta_path}")


if __name__ == "__main__":
    main()
