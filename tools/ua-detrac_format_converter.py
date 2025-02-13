import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import concurrent.futures

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def extract_uadetract_xml(object):
    """
        The ua-detrac objects
    Args:
        - obj: The xml format
    """
    pass


def save_dict_list_to_json(data, file_path):
  """Saves a list of dictionaries to a JSON file.

  Args:
    data_list: A list of dictionaries to be saved.
    filename: The name of the JSON file to be created.
  """

  with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)

  logger.info(f"Write to {file_path} successfully")


def parse_opt():
    parser = argparse.ArgumentParser(prog="ua-detrac_format_converter.py")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="./annotation_folder",
        help="The input xml folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./annotation_json.json",
        help="The output json annotation file after transformation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number concurrently workers",
    )
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <annotation_file>")
    #     sys.exit(1)
    #
    # opt = parse_opt()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     executor.map(extract_uadetract_xml, range(3))

    folder_path = Path("/home/tienpm/Documents/PROJECT/tensorflow-image-detection/.cache/DETRAC-Train-Annotations-XML")
    # for file in folder_path.glob("*.xml"):
    #     print(file.absolute())

    in_path = folder_path.joinpath("MVI_20011.xml")

    tree = ET.parse(in_path)
    root = tree.getroot()

    for item in root.findall('frame'):
        print(f"len: {len(item)}")
        print(item.attrib.keys())
        print(item.attrib["density"])
        print(item[0])
        for child in item:
            print(len(child))
            break
        break

