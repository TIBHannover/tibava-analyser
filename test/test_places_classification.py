import sys
import argparse
import logging

from analyser.client import AnalyserClient


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--input_path", default="/media/test.mp4", help="path to input video .mp4")
    parser.add_argument("--output_path", default="/media", help="path to output folder")
    parser.add_argument("--shots", action="store_true", help="aggregate labels by detected shots")
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
    data_id = client.upload_data(args.input_path)
    logging.info(f"Upload done: {data_id}")

    """
    Run shot detection if specified
    """
    shots_id = None
    shots = None
    if args.shots:
        job_id = client.run_plugin("transnet_shotdetection", [{"id": data_id, "name": "video"}], [])
        logging.info(f"Job transnet_shotdetection started: {job_id}")

        result = client.get_plugin_results(job_id=job_id)
        if result is None:
            logging.error("Job is crashing")
            return

        for output in result.outputs:
            if output.name == "shots":
                shots_id = output.id

        shots = client.download_data(shots_id, args.output_path)

    """
    Run place classification
    """
    job_id = client.run_plugin("places_classifier", [{"id": data_id, "name": "video"}], [])
    logging.info(f"Job places_classifier started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    logging.info(result)

    embeddings_id = None
    places365_id = None
    places16_id = None
    places3_id = None

    for output in result.outputs:
        if output.name == "embeddings":
            embeddings_id = output.id
        if output.name == "probs_places365":
            places365_id = output.id
        if output.name == "probs_places16":
            places16_id = output.id
        if output.name == "probs_places3":
            places3_id = output.id

    logging.info("#### Embeddings")
    logging.info(embeddings_id)
    logging.info(client.download_data(embeddings_id, args.output_path))

    logging.info("#### Places365 results")
    logging.info(places365_id)
    logging.info(client.download_data(places365_id, args.output_path))

    logging.info("#### Places16 results")
    logging.info(places16_id)
    logging.info(client.download_data(places16_id, args.output_path))

    logging.info("#### Places3 results")
    logging.info(places3_id)
    logging.info(client.download_data(places3_id, args.output_path))

    """
    Aggregate places label by shot
    """
    if args.shots and shots:
        job_id = client.run_plugin(
            "shot_annotator", [{"id": shots_id, "name": "shots"}, {"id": places3_id, "name": "probs"}], []
        )
        logging.info(f"Job shot_annotator started: {job_id}")

        result = client.get_plugin_results(job_id=job_id)
        if result is None:
            logging.error("Job is crashing")
            return
        logging.info(result)

        annotation_id = None
        for output in result.outputs:
            if output.name == "annotations":
                annotation_id = output.id

        logging.info("#### Places3 annotations by shot")
        logging.info(annotation_id)
        logging.info(client.download_data(annotation_id, args.output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())