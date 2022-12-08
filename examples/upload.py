import sys
import argparse
import logging

from analyser.client import AnalyserClient
from analyser.data import ShotsData, Shot, ImageData, ImagesData, DataManager
from analyser.data import generate_id, create_data_path
import numpy as np
import imageio.v3 as iio


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--input_path", default="/media/test.mp4", help="path to input video .mp4")
    parser.add_argument("--output_path", default="/media", help="path to output folder")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    client = AnalyserClient("localhost", 50051)
    logging.info(f"Start uploading")
    data_id = client.upload_file(args.input_path)
    logging.info(f"Upload done: {data_id}")

    exists = client.check_data(data_id)
    print(exists)

    # logging.info(f"Downloading: {data_id}")
    # logging.info(client.download_data(data_id, args.output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())