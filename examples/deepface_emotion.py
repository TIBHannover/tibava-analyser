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
    job_id = client.run_plugin("insightface_video_detector", [{"id": data_id, "name": "video"}], [])
    logging.info(f"Job insightface_video_detector started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    # get facial images from facedetection
    images_id = None
    for output in result.outputs:
        if output.name == "images":
            images_id = output.id

    # deepface_emotion
    job_id = client.run_plugin("deepface_emotion", [{"id": images_id, "name": "images"}], [])
    logging.info(f"Job deepface_emotion started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    # get emotions
    output_id = None
    for output in result.outputs:
        if output.name == "probs":
            output_id = output.id

    logging.info(client.download_data(output_id, args.output_path))

    # aggregate emotions per frame (multiple faces and thus emotions are depicted)
    job_id = client.run_plugin("aggregate_scalar_per_time", [{"id": output_id, "name": "timeline"}], [])
    logging.info(f"Job aggregate_scalar_per_time started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    output_id = None
    for output in result.outputs:
        if output.name == "aggregated_timeline":
            output_id = output.id

    logging.info(client.download_data(output_id, args.output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())