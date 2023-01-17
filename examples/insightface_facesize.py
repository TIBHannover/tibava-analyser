import argparse
import logging
import sys

from analyser.client import AnalyserClient


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

    # insightface_detection
    job_id = client.run_plugin("insightface_video_detector_torch", [{"id": data_id, "name": "video"}], [])
    logging.info(f"Job insightface_video_detector_torch started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    # get facial images from facedetection
    bboxes_id = None
    for output in result.outputs:
        if output.name == "bboxes":
            bboxes_id = output.id

    # facesizes
    # bboxes_id = "13d6f37f9b31473aad68f1ad1fee96a6"
    job_id = client.run_plugin("insightface_facesize", [{"id": bboxes_id, "name": "bboxes"}], [])
    logging.info(f"Job insightface_facesize started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    # get shot size prediction
    output_id_probs = None
    output_id_facesizes = None
    for output in result.outputs:
        if output.name == "probs":
            output_id_probs = output.id
        if output.name == "facesizes":
            output_id_facesizes = output.id
    data = client.download_data(output_id_probs, args.output_path)
    with data:
        logging.info(data)

    data = client.download_data(output_id_facesizes, args.output_path)
    with data:
        logging.info(data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
