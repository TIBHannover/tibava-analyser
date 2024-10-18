from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager  # type: ignore
from analyser.data import AudioData, AnnotationData, Annotation  # type: ignore

from analyser.data import DataManager, Data  # type: ignore

from typing import Callable, Dict, Any, Tuple
import logging

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"language": "de"}

requires = {
    "audio": AudioData,
}

provides = {
    "annotations": AnnotationData,
}

def get_speaker_turns(speaker_segments, gap: float=0.01) -> Dict[str, list[Annotation]]:
    speaker_turns = []
    speakers = {}

    last_segment = None
    for segment in sorted(speaker_segments, key=lambda x: x["start"]):
        current_segment = {
            "start": segment["start"],
            "end": segment["end"],
            "labels": segment.get("text", "").strip(),
            "speaker": segment.get("speaker", "Unknown"),
        }
        if last_segment:
            if last_segment["speaker"] == current_segment["speaker"]:
                last_segment["end"] = current_segment["end"]
                last_segment["labels"] += " " + current_segment["labels"]
            else:
                if current_segment["start"] - last_segment["end"] <= gap:
                    current_segment["start"] = last_segment["end"] + gap
                speaker_turns.append(current_segment)
                
        last_segment = current_segment
    
    for turn in speaker_turns:
        if not turn["speaker"] in speakers.keys():
            speakers[turn["speaker"]] = []
        speakers[turn["speaker"]].append(Annotation(start=turn["start"], end=turn["end"], labels=turn["labels"]))
    
    return speakers

def get_model(device: str, config: Dict[str, Any]) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    import whisperx  # type: ignore
    model = whisperx.load_model("large-v3", device=device, compute_type="float16", language=config["language"])
    diarize_model = whisperx.DiarizationPipeline(device=device)#use_auth_token=config['huggingface']['token'], device=device)
    alignment_model, metadata = whisperx.load_align_model(language_code="de", device=device)
    return model, diarize_model, alignment_model, metadata

@AnalyserPluginManager.export("whisper_x")
class WhisperX(
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

        self.model = None
        self.model_name = self.config.get("model", "whisper_x")

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        import librosa  # type: ignore
        import torch
        import whisperx  # type: ignore

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.model is None:
            self.model, self.diarize_model, self.alignment_model, self.metadata = get_model(
                config=parameters, # TODO
                device=device,
            )
            self.device = device

        with inputs["audio"] as input_data, data_manager.create_data(
            "ListData"
        ) as output_data:
            with input_data.open_audio("r") as f_audio:
                y, sr = librosa.load(f_audio, sr=16000)
                transcription = self.model.transcribe(y, 8, parameters.get("language"))
                aligned_transcription = whisperx.align(transcription["segments"], self.alignment_model, self.metadata, y, device, return_char_alignments=False)
                #aligned_segments = aligned_transcription["segments"]

                diarize_segments = self.diarize_model(y)
                speaker_transcription = whisperx.assign_word_speakers(diarize_segments, aligned_transcription)
                speaker_turns = get_speaker_turns(speaker_transcription["segments"])

                for speaker, annotations in speaker_turns.items():
                    with output_data.create_data("AnnotationData") as ann_data:
                        ann_data.annotations.extend(annotations)
                        ann_data.name = speaker

                self.update_callbacks(callbacks, progress=1.0)
                return {"annotations": output_data}
