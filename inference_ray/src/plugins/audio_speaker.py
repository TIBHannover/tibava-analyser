from typing import List, Any, Tuple

from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager  # type: ignore
from analyser.data import AudioData, AnnotationData, ListData, Annotation  # type: ignore

from analyser.data import DataManager, Data  # type: ignore

from typing import Callable, Dict
import logging

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {}

requires = {
    "audio": AudioData,
    "annotations": ListData,
}

provides = {"gender_annotations": ListData, "emotion_annotations": ListData}


@AnalyserPluginManager.export("audio_speaker_analysis")
class AudioSpeakerAnalysis(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # inference_config = self.config.get("inference", None)
        # self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

        self.emotion_model = None
        self.gender_model = None
        self.gender_processor = None

        self.emotion_label_map: Dict[str, str] = {
            "neu": "Neutral",
            "hap": "Happy",
            "ang": "Angry",
            "sad": "Sad",
        }
        self.gender_label_map: Dict[str, str] = {"0": "Female", "1": "Male"}

        self.model_name = self.config.get("model", "speaker_attribute_model")

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        import librosa
        import torch
        import numpy as np
        from speechbrain.inference.interfaces import foreign_class
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2ForSequenceClassification,
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        def get_models() -> Tuple[Any, Any, Any]:
            run_opts = {"device": device}
            emo_model = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                run_opts=run_opts,
            )
            gen_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
            )
            gen_proc = Wav2Vec2FeatureExtractor.from_pretrained(
                "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
            )
            #TODO change model saving directory
            gen_model.to(device)
            gen_model.eval()
            return emo_model, gen_model, gen_proc

        def classify_segments(
            audio_array: np.ndarray,
            speaker_turns: List[Dict[str, Any]],
            sampling_rate: int,
        ) -> Tuple[List[Annotation], List[Annotation]]:
            """
            Takes Speaker Diarization segments and classifies them into speech emotion and gender categories.

            Args:
                audio_array (np.ndarray): Loaded audio series
                speaker_turns (List[Dict[str, Any]]): List of speaker segments from ASR
                sampling_rate (int): Sampling rate of audio_array

            Returns:
                Tuple[List[Annotation], List[Annotation]]: List of segment gender and emotion predictions
            """
            gender_predictions = []
            emotion_predictions = []

            for seg in speaker_turns:
                # slice audio of current segment
                segment_boundaries = librosa.time_to_samples(
                    np.array([seg.start, seg.end]), sr=sampling_rate
                )
                seg_audio_array = audio_array[
                    segment_boundaries[0] : segment_boundaries[1]
                ]
                seg_audio_tensor = (
                    torch.tensor(seg_audio_array).unsqueeze(0).to(torch.float32)
                )

                if seg_audio_tensor.shape[1] < sampling_rate:
                    seg_audio_tensor = torch.nn.functional.pad(
                        seg_audio_tensor, (0, sampling_rate - seg_audio_tensor.shape[1])
                    )

                ## Chop audio into further segments of 10 seconds if audio is longer than 10 seconds
                if seg_audio_tensor.shape[1] > sampling_rate * 10:
                    ceiling_len = (
                        seg_audio_tensor.shape[1] // (sampling_rate * 10)
                    ) * (sampling_rate * 10)
                    audio_segments = torch.tensor(
                        np.array(
                            [
                                seg_audio_tensor[:, i : i + sampling_rate * 10]
                                for i in range(0, ceiling_len, sampling_rate * 10)
                            ]
                        )
                    ).squeeze(
                        1
                    )  ## --> (N, 1600000)
                else:
                    audio_segments = seg_audio_tensor

                audio_segments = audio_segments.to(device)

                input_values = self.gender_processor(
                    audio_segments, sampling_rate=sampling_rate, return_tensors="pt"
                ).input_values.squeeze(
                    0
                )  ## --> (N, 1600000)
                input_values = input_values.to(device)

                with torch.no_grad():
                    result = self.gender_model(input_values).logits.softmax(dim=1)
                    sum_res = result.mean(dim=0)
                    max_i = sum_res.detach().cpu().argmax().item()
                    gen_prob = sum_res.detach().cpu().max().item()
                    gen_pred = self.gender_label_map[str(max_i)]

                    _, emo_prob, _, text_lab = self.emotion_model.classify_batch(
                        audio_segments
                    )

                if len(text_lab) > 1:
                    top_probs = (
                        emo_prob.detach()
                        .cpu()
                        .topk(min(len(text_lab), 3))
                        .values.tolist()
                    )
                    top_preds = [
                        self.emotion_label_map[text_lab[i]]
                        for i in emo_prob.detach()
                        .cpu()
                        .topk(min(len(text_lab), 3))
                        .indices.tolist()
                    ]
                else:
                    top_probs = [emo_prob.item()]
                    top_preds = [self.emotion_label_map[text_lab[0]]]

                gender_predictions.append(
                    Annotation(
                        start=seg.start,
                        end=seg.end,
                        labels=[
                            {
                                "gender_pred": gen_pred,
                                "gender_prob": gen_prob,
                            }
                        ],
                    )
                )
                emotion_predictions.append(
                    Annotation(
                        start=seg.start,
                        end=seg.end,
                        labels=[
                            {
                                "emotion_pred_top3": top_preds,
                                "emotion_prob_top3": top_probs,
                                "emotion_pred": top_preds[0],
                            }
                        ],
                    )
                )

            return gender_predictions, emotion_predictions

        if None in [self.emotion_model, self.gender_model, self.gender_processor]:
            self.emotion_model, self.gender_model, self.gender_processor = get_models()

        with inputs["audio"] as input_audio, inputs[
            "annotations"
        ] as input_annotations, data_manager.create_data(
            "ListData"
        ) as gender_output_data, data_manager.create_data(
            "ListData"
        ) as emotion_output_data:
            with input_audio.open_audio("r") as audio_file:
                sampling_rate = 16000
                audio_array, _ = librosa.load(audio_file, sr=sampling_rate)
                for _, speaker_data in input_annotations:
                    with speaker_data as speaker_data:
                        gender_predictions, emotion_predictions = classify_segments(
                            audio_array, speaker_data.annotations, sampling_rate
                        )

                        with gender_output_data.create_data(
                            "AnnotationData"
                        ) as gender_ann_data:
                            gender_ann_data.annotations.extend(gender_predictions)
                            gender_ann_data.name = speaker_data.name
                        with emotion_output_data.create_data(
                            "AnnotationData"
                        ) as emotion_ann_data:
                            emotion_ann_data.annotations.extend(emotion_predictions)
                            emotion_ann_data.name = speaker_data.name

                self.update_callbacks(callbacks, progress=1.0)
                return {
                    "gender_annotations": gender_output_data,
                    "emotion_annotations": emotion_output_data,
                }
