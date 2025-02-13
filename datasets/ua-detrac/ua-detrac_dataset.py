import csv
import json
import os

import gdown
import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{10.1016/j.cviu.2020.102907,
author = {Wen, Longyin and Du, Dawei and Cai, Zhaowei and Lei, Zhen and Chang, Ming-Ching and Qi, Honggang and Lim, Jongwoo and Yang, Ming-Hsuan and Lyu, Siwei},
title = {UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking},
year = {2020},
issue_date = {Apr 2020},
publisher = {Elsevier Science Inc.},
address = {USA},
volume = {193},
number = {C},
issn = {1077-3142},
url = {https://doi.org/10.1016/j.cviu.2020.102907},
doi = {10.1016/j.cviu.2020.102907},
journal = {Comput. Vis. Image Underst.},
month = apr,
numpages = {20},
keywords = {Evaluation protocol, Benchmark, Object tracking, Object detection}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
The University at Albany DEtection and TRACking (UA-DETRAC) dataset for comprehensive performance evaluation of MOT systems especially on detectors. 
The UA-DETRAC benchmark dataset consists of 100 challenging videos captured from real-world trac scenes (over 140,000 frames with rich annotations, including illumination, vehicle type, occlusion, truncation ratio, and vehicle bounding boxes) for multi-object detection and tracking.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://sites.google.com/view/daweidu/projects/ua-detrac"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "images": "https://drive.google.com/uc?id=1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm",
    "train_annotations": "https://drive.google.com/uc?id=1zc8KZWUa3Cw0DbcFqTCMBmUZzbfuhGKr",
    "test_annotation": "https://drive.google.com/uc?id=1f22g3puG164gemY_njcPG8beuCcuMjqb"

}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class UADETRACDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="train", version=VERSION, description="This training part of UA-DETRAC dataset"),
        datasets.BuilderConfig(name="test", version=VERSION, description="This testing part of UA-DETRAC dataset"),
    ]

    DEFAULT_CONFIG_NAME = "train"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "train":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "image": datasets.Image(),
                    "objects": datasets.Sequence(features={"track_id": datasets.Value("int32"), "bbox": datasets.Array(shape=(1, 4), dtype="float32"),}),
                    "scene_weather": datasets.Value("string"),

                    # These are the features of your dataset like images, labels ...
                }
            )
        else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {

                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "first_domain":
                    # Yields examples as (key, example) tuples
                    yield key, {
                        "sentence": data["sentence"],
                        "option1": data["option1"],
                        "answer": "" if split == "test" else data["answer"],
                    }
                else:
                    yield key, {
                        "sentence": data["sentence"],
                        "option2": data["option2"],
                        "second_domain_answer": "" if split == "test" else data["second_domain_answer"],
                    }