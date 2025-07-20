import datasets
import json
import os
from PIL import Image

# Define dataset configuration
_VERSION = datasets.Version("2.0.0")

_DESCRIPTION = """\
The COCO (Common Objects in Context) dataset is a large-scale object detection, segmentation, and captioning dataset.
This script provides access to the 2017 release of the COCO dataset, including:
- Train2017 images and instances/captions annotations.
- Val2017 images and instances/captions annotations.
- Test2017 images (no instances/captions annotations released for test set).
"""

_HOMEPAGE = "[https://cocodataset.org/](https://cocodataset.org/)"

_LICENSE = """\
Creative Commons Attribution 4.0 International Public License (CC BY 4.0)
[https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
"""

_CITATION = """\
@inproceedings{lin2014microsoft,
  title={{Microsoft COCO: Common Objects in Context}},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European Conference on Computer Vision},
  pages={740-755},
  year={2014},
  organization={Springer}
}
"""

_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "test_images": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations_trainval": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "image_info_test": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
}


class Coco2017(datasets.GeneratorBasedBuilder):
    """COCO 2017 Dataset."""

    VERSION = _VERSION

    def _info(self):
        # Define the features of the dataset
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "image_id": datasets.Value("int32"),
                    "image": datasets.Image(),  # Image feature to handle image loading
                    "width": datasets.Value("int32"),
                    "height": datasets.Value("int32"),
                    "file_name": datasets.Value("string"),
                    "annotations": datasets.Sequence(
                        {
                            "id": datasets.Value("int32"),
                            "bbox": datasets.Sequence(
                                datasets.Value("float32")
                            ),  # [x, y, width, height]
                            "area": datasets.Value("float32"),
                            "category_id": datasets.Value("int32"),
                            "iscrowd": datasets.Value("bool"),
                            # Segmentation can be very complex (RLE or polygons).
                            # For simplicity, we'll store it as a string or list of lists.
                            # Here, we will store it as a list of lists of floats for polygons,
                            # and potentially a string for RLE (though RLE is less common for direct use).
                            "segmentation": datasets.Sequence(
                                datasets.Sequence(datasets.Value("float32"))
                            ),
                        }
                    ),
                    "captions": datasets.Sequence(
                        {
                            "id": datasets.Value("int32"),
                            "caption": datasets.Value("string"),
                        }
                    ),
                    # We can also add category names if needed for convenience
                    "category_names": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=("image", "annotations"),
        )

    def _split_generators(self, dl_manager):
        # Download and extract the dataset files
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        print(f"DEBUG: {downloaded_files}")

        # Paths to extracted images and annotation JSONs
        train_image_dir = os.path.join(downloaded_files["train_images"], "train2017")
        val_image_dir = os.path.join(downloaded_files["val_images"], "val2017")
        test_image_dir = os.path.join(downloaded_files["test_images"], "test2017")

        annotations_path = downloaded_files["annotations_trainval"]
        image_info_test_path = downloaded_files["image_info_test"]

        # Define splits
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_dir": train_image_dir,
                    "instances_file": os.path.join(
                        annotations_path, "annotations", "instances_train2017.json"
                    ),
                    "captions_file": os.path.join(
                        annotations_path, "annotations", "captions_train2017.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "image_dir": val_image_dir,
                    "instances_file": os.path.join(
                        annotations_path, "annotations", "instances_val2017.json"
                    ),
                    "captions_file": os.path.join(
                        annotations_path, "annotations", "captions_val2017.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_dir": test_image_dir,
                    "image_info_file": os.path.join(
                        image_info_test_path, "annotations", "image_info_test2017.json"
                    ),
                    # Test split does not have instance/caption annotations
                    "instances_file": None,
                    "captions_file": None,
                },
            ),
        ]

    def _generate_examples(
        self, image_dir, instances_file=None, captions_file=None, image_info_file=None
    ):
        """Yields examples as (key, example) tuples."""

        # Load instances annotations
        instances_data = {"images": [], "annotations": [], "categories": []}
        if instances_file and os.path.exists(instances_file):
            with open(instances_file, "r", encoding="utf-8") as f:
                instances_data = json.load(f)

        # Load captions annotations
        captions_data = {"annotations": []}
        if captions_file and os.path.exists(captions_file):
            with open(captions_file, "r", encoding="utf-8") as f:
                captions_data = json.load(f)

        # Load image info for test split (if instances_file is None)
        image_info_data = {"images": []}
        if image_info_file and os.path.exists(image_info_file):
            with open(image_info_file, "r", encoding="utf-8") as f:
                image_info_data = json.load(f)

        # Create mapping from image_id to its properties
        # For train/val, use images from instances_data. For test, use image_info_data.
        images_info = {}
        if instances_data["images"]:  # train/val
            for img in instances_data["images"]:
                images_info[img["id"]] = img
        elif image_info_data["images"]:  # test
            for img in image_info_data["images"]:
                images_info[img["id"]] = img

        # Map image_id to its annotations
        image_annotations = {}
        for ann in instances_data["annotations"]:
            image_annotations.setdefault(ann["image_id"], []).append(ann)

        # Map image_id to its captions
        image_captions = {}
        for cap in captions_data["annotations"]:
            image_captions.setdefault(cap["image_id"], []).append(cap)

        # Map category_id to category name
        category_id_to_name = {
            cat["id"]: cat["name"] for cat in instances_data["categories"]
        }

        # Iterate through images and yield examples
        # We iterate through the image files on disk to ensure all images are processed,
        # including those that might not have annotations (e.g., in the test set or rare cases).
        for img_file_name in os.listdir(image_dir):
            if not img_file_name.endswith((".jpg", ".jpeg", ".png")):
                continue

            # COCO image IDs are typically integers, derived from file names like '000000123456.jpg'
            try:
                # Extract image_id from file_name (e.g., '000000123456.jpg' -> 123456)
                # This assumes COCO's standard file naming convention.
                image_id = int(os.path.splitext(img_file_name)[0])
            except ValueError:
                # Skip files that don't conform to the expected naming convention
                print(f"Warning: Skipping unexpected file name: {img_file_name}")
                continue

            image_path = os.path.join(image_dir, img_file_name)

            # Get image metadata from our pre-parsed info
            img_info = images_info.get(image_id)
            if not img_info:
                print(
                    f"Warning: No metadata found for image ID {image_id} ({img_file_name}). Skipping."
                )
                continue

            # Load image using PIL for datasets.Image() feature
            try:
                image_obj = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping.")
                continue

            # Get annotations and captions for the current image
            current_image_annotations = image_annotations.get(image_id, [])
            current_image_captions = image_captions.get(image_id, [])

            # Prepare segmentation for the Hugging Face dataset feature
            processed_annotations = []
            current_category_names = []

            for ann in current_image_annotations:
                segmentation_data = ann.get("segmentation", [])
                # Ensure segmentation is a list of lists of floats for polygons.
                # If RLE (dict), you might choose to skip or convert it if possible.
                if isinstance(
                    segmentation_data, dict
                ):  # Handle RLE, which is usually a dictionary.
                    # For simplicity, we'll store empty list or handle conversion if needed.
                    segmentation_data = (
                        []
                    )  # Or you can add logic to convert RLE to polygons if a library supports it.
                elif not isinstance(segmentation_data, list):
                    segmentation_data = []  # Ensure it's a list.

                processed_annotations.append(
                    {
                        "id": ann["id"],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "category_id": ann["category_id"],
                        "iscrowd": bool(ann.get("iscrowd", 0)),  # Ensure boolean type
                        "segmentation": segmentation_data,
                    }
                )

                # Collect category names for convenience
                if ann["category_id"] in category_id_to_name:
                    current_category_names.append(
                        category_id_to_name[ann["category_id"]]
                    )
                else:
                    current_category_names.append(
                        "unknown"
                    )  # Fallback for categories not found

            # Yield the example
            yield image_id, {
                "image_id": image_id,
                "image": image_obj,
                "width": img_info["width"],
                "height": img_info["height"],
                "file_name": img_info["file_name"],
                "annotations": processed_annotations,
                "captions": [
                    {"id": cap["id"], "caption": cap["caption"]}
                    for cap in current_image_captions
                ],
                "category_names": list(
                    set(current_category_names)
                ),  # Unique category names for the image
            }
