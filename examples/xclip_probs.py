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

    # generate image embeddings
    job_id = client.run_plugin("x_clip_video_embedding", [{"id": data_id, "name": "video"}], [])
    logging.info(f"Job x_clip_video_embedding started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    embd_id = None
    image_features_id = None
    video_features_id = None
    for output in result.outputs:
        if output.name == "image_features":
            image_features_id = output.id
        if output.name == "video_features":
            video_features_id = output.id
    logging.info(f"finished job with resulting embedding id: {embd_id}")
    # calculate similarities between image embeddings and search term
    job_id = client.run_plugin(
        "x_clip_probs",
        [{"id": image_features_id, "name": "image_features"}, {"id": video_features_id, "name": "video_features"}],
        [{"name": "search_term", "value": "This is a test."}],
    )
    logging.info(f"Job clip_probs started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    probs_id = None
    for output in result.outputs:
        if output.name == "probs":
            probs_id = output.id
    logging.info(client.download_data(probs_id, args.output_path))
    logging.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())