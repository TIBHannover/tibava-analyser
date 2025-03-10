from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager
from analyser.data import BboxesData, AnnotationData, Annotation, ShotsData, FacesData

from analyser.data import DataManager, Data

import logging

import numpy as np
import os
import subprocess
from typing import Callable, Any, Dict, List, Optional
from collections import defaultdict

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {
    "fps": 25,  # detected automatically by bbox delta_time
    "min_track": 1,  # in seconds instead of frames (25)
    "num_failed_det": 10,
    "min_face_size": 0,  # TODO adapt min_face_size check to normalized bboxes, hard constrained already included: bbox.h > 0.05
    "crop_scale": 0.4,
    "workers": 4,
}

requires = {
    "bboxes": BboxesData,
    "faces": FacesData, # TODO does not contain any information? remove
    "shots": ShotsData,
}

provides = {
    "track_data": AnnotationData,
}
""" Annotation label objects:
{
"frames": frame indices
"bboxes": normalized x1y1x2y2 format #TODO change back to normalized xywh format for compatibility with usual tibava bboxes?
"track_id": str
}
"""


@AnalyserPluginManager.export("face_tracking")
class FaceTracker(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        with inputs["bboxes"] as bbox_data, inputs["faces"] as faces_data, inputs[
            "shots"
        ] as shots_data:
            parameters["fps"] = 1 / bbox_data.bboxes[0].delta_time

            faces_by_frame = defaultdict(list)
            for face, bbox in zip(faces_data.faces, bbox_data.bboxes):
                if bbox.det_score >= 0.6 and bbox.h > 0.05:
                    faces_by_frame[bbox.time].append(
                        {
                            "id": face.id,
                            "frame": int(
                                bbox.time * parameters["fps"]
                            ),  # TODO maybe get frame id directly from somewhere (face.ref_id does not work)
                            "bbox": self.convert_bbox(bbox),
                            "conf": bbox.det_score,
                        }
                    )

            max_frame = len(faces_by_frame.keys())
            faces_list = [[] for _ in range(max_frame + 1)]
            for frame_index, face_list in enumerate(faces_by_frame.values()):
                faces_list[frame_index] = face_list

            # TODO change to bbox data type?
            with data_manager.create_data("AnnotationData") as track_data:
                for shot in shots_data.shots:
                    if shot.end - shot.start >= parameters.get("min_track"):
                        tracks = self.track_shot(
                            parameters,
                            faces_list[
                                int(shot.start * parameters["fps"]) : int(
                                    shot.end * parameters["fps"]
                                )
                            ],
                        )
                        for track in tracks:
                            annotation = Annotation(
                                start=float(track["frame"][0]) / parameters["fps"],
                                end=float(track["frame"][-1]) / parameters["fps"],
                                labels=[
                                    {
                                        "frames": track["frame"].tolist(),
                                        "bboxes": track["bbox"].tolist(),
                                        "track_id": track["track_id"],
                                    }
                                ],
                            )
                            track_data.annotations.append(annotation)

                return {
                    "track_data": track_data,
                }

    def bb_intersection_over_union(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def convert_bbox(self, bbox) -> List[int]:
        _bbox = bbox.to_dict()
        x, y, w, h = _bbox["x"], _bbox["y"], _bbox["w"], _bbox["h"]
        return [x, y, x + w, y + h]

    def normalize_to_pixel(
        self, bbox: Dict[str, float], frame_width: int, frame_height: int
    ) -> List[float]:
        """unused"""
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        x1 = x * frame_width
        y1 = y * frame_height
        x2 = (x + w) * frame_width
        y2 = (y + h) * frame_height
        return [x1, y1, x2, y2]

    def track_shot(
        self, params: Dict[str, Any], shotFaces: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        from scipy.interpolate import interp1d
        import uuid

        iouThres = 0.5
        tracks = []
        while True:
            track = []
            enhanced_track = {
                "frame": [],
                "bbox": [],
            }
            for frameFaces in shotFaces:
                for face in frameFaces:
                    if not track:
                        track.append(face)
                        frameFaces.remove(face)
                        enhanced_track["frame"].append(face["frame"])
                        enhanced_track["bbox"].append(face["bbox"])
                    elif face["frame"] - track[-1]["frame"] <= params["num_failed_det"]:
                        iou = self.bb_intersection_over_union(
                            face["bbox"], track[-1]["bbox"]
                        )
                        if iou > iouThres:
                            track.append(face)
                            frameFaces.remove(face)
                            enhanced_track["frame"].append(face["frame"])
                            enhanced_track["bbox"].append(face["bbox"])
                            continue
                    else:
                        break
            if not track:
                break
            elif len(track) > params["min_track"]:
                frameNum = np.array(enhanced_track["frame"])
                bboxes = np.array(enhanced_track["bbox"])
                frameI = np.arange(frameNum[0], frameNum[-1] + 1)
                bboxesI = []
                for ij in range(0, 4):
                    interpfn = interp1d(frameNum, bboxes[:, ij])
                    bboxesI.append(interpfn(frameI))
                bboxesI = np.stack(bboxesI, axis=1)
                if (
                    max(
                        np.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                        np.mean(bboxesI[:, 3] - bboxesI[:, 1]),
                    )
                    > params["min_face_size"]
                ):
                    track_id = str(uuid.uuid4())
                    enhanced_track["track_id"] = track_id
                    enhanced_track["frame"] = frameI
                    enhanced_track["bbox"] = bboxesI

                    tracks.append(enhanced_track)
        return tracks

    def crop_video(
        self,
        params: Dict[str, Any],
        track: Dict[str, Any],
        vr: List[np.ndarray],
        fps: float,
        cropFile: str,
        audioFilePath: Optional[str] = None,
    ) -> str:
        """
        unused
        Important: expects bboxes in x1y1x2y2 format

        returns video file path
        """
        from scipy import signal
        import cv2

        fp = cropFile + ".avi"
        if audioFilePath:
            fp = cropFile + "t.avi"
        vOut = cv2.VideoWriter(fp, cv2.VideoWriter_fourcc(*"XVID"), fps, (224, 224))

        dets = {"x": [], "y": [], "s": []}
        for det in track["bbox"]:
            dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
            dets["y"].append((det[1] + det[3]) / 2)
            dets["x"].append((det[0] + det[2]) / 2)

        dets["s"] = np.array(signal.medfilt(dets["s"], kernel_size=13))
        dets["x"] = np.array(signal.medfilt(dets["x"], kernel_size=13))
        dets["y"] = np.array(signal.medfilt(dets["y"], kernel_size=13))

        cs = params["crop_scale"]
        frame_nums = np.array(track["frame"])

        for fidx, frame_num in enumerate(frame_nums):
            image = cv2.cvtColor(vr[frame_num], cv2.COLOR_RGB2BGR)

            bs = dets["s"][fidx]
            my = dets["y"][fidx]
            mx = dets["x"][fidx]

            y1 = int(my - bs)
            y2 = int(my + bs * (1 + 2 * cs))
            x1 = int(mx - bs * (1 + cs))
            x2 = int(mx + bs * (1 + cs))

            pad_top = max(0, -y1)
            pad_bottom = max(0, y2 - image.shape[0])
            pad_left = max(0, -x1)
            pad_right = max(0, x2 - image.shape[1])

            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                image = cv2.copyMakeBorder(
                    image,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=[110, 110, 110],
                )

            crop_y1 = max(0, y1 + pad_top)
            crop_y2 = min(image.shape[0], y2 + pad_top)
            crop_x1 = max(0, x1 + pad_left)
            crop_x2 = min(image.shape[1], x2 + pad_left)

            face = image[crop_y1:crop_y2, crop_x1:crop_x2]
            face_resized = cv2.resize(face, (224, 224))

            vOut.write(face_resized)

        vOut.release()
        if audioFilePath:
            subaudioFilePath = cropFile + ".wav"
            audioStart = frame_nums[0] / fps
            audioEnd = (frame_nums[-1] + 1) / fps

            command = (
                f"ffmpeg -y -i {audioFilePath} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads {params['workers']} "
                f"-ss {audioStart:.3f} -to {audioEnd:.3f} {subaudioFilePath} -loglevel panic"
            )
            subprocess.call(command, shell=True)

            command = (
                f"ffmpeg -y -i {cropFile}t.avi -threads {params['workers']} "  # -i {subaudioFilePath} removed
                f"-c:v copy -c:a copy {cropFile}.avi -loglevel panic"
            )
            subprocess.call(command, shell=True)

            os.remove(cropFile + "t.avi")

        return cropFile + ".avi"
