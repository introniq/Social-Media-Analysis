import os, json
from pathlib import Path
import ffmpeg
import torch
from PIL import Image
import pytesseract

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

import whisper
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from ultralytics import YOLO


# ------------------------
# Utils
# ------------------------

def extract_audio(video, out_audio):
    (
        ffmpeg.input(video)
              .output(out_audio, ac=1, ar="16000", format="wav")
              .overwrite_output()
              .run(quiet=True)
    )
    return out_audio


def transcribe(audio, model="small"):
    w = whisper.load_model(model)
    r = w.transcribe(audio)
    return r["segments"]


def detect_scenes(video):
    vm = VideoManager([video])
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    base = vm.get_base_timecode()

    vm.start()
    sm.detect_scenes(frame_source=vm)
    scenes = sm.get_scene_list(base)
    vm.release()

    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


def extract_frame(video, timestamp, out_path):
    (
        ffmpeg.input(video, ss=timestamp)
              .output(out_path, vframes=1)
              .overwrite_output()
              .run(quiet=True)
    )


def caption_image(img, processor, model, device):
    inp = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inp)
    return processor.decode(out[0], skip_special_tokens=True)


def ocr(img):
    return pytesseract.image_to_string(img).strip()


def build_scene_text(scene, transcript, caption, objects, ocr_text):
    return f"""
SCENE {scene}:
Caption: {caption}
Objects: {objects}
OCR: {ocr_text}

Transcript:
{transcript}
"""


def generate_story_free(scene_texts, generator):
    story = ""
    for txt in scene_texts:
        out = generator(
            f"Convert this into a clear story scene with dialogues if present:\n\n{txt}\n\nStory:",
            max_length=350
        )[0]["generated_text"]
        story += out + "\n\n"
    return story


# ------------------------
# Main
# ------------------------

def video_to_story_free(video, out="story_out"):
    os.makedirs(out, exist_ok=True)
    base = Path(out)

    # 1. Scenes
    scenes = detect_scenes(video)
    print("Scenes:", len(scenes))

    # 2. Audio + Whisper
    audio = str(base / "audio.wav")
    extract_audio(video, audio)
    segs = transcribe(audio)

    # 3. Models (all free)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    yolo = YOLO("yolov8n.pt")
    t5 = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)

    scene_texts = []

    for i, (start, end) in enumerate(scenes):
        mid = (start + end) / 2
        frame_path = str(base / f"frame_{i}.jpg")
        extract_frame(video, mid, frame_path)

        img = Image.open(frame_path).convert("RGB")

        # Captioning
        cap = caption_image(img, processor, blip, device)

        # OCR
        ocr_text = ocr(img)

        # Object detection
        det = yolo(frame_path, verbose=False)[0]
        objs = [det.names[int(c.cls)] for c in det.boxes]

        # Transcript for scene
        t = "\n".join([x["text"] for x in segs if not (x["end"] < start or x["start"] > end)])

        scene_texts.append(
            build_scene_text(i, t, cap, objs, ocr_text)
        )

    # Story from free model
    story = generate_story_free(scene_texts, t5)

    with open(base / "story.txt", "w", encoding="utf-8") as f:
        f.write(story)

    print("Saved:", base / "story.txt")
    return story

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="story_out")
    args = parser.parse_args()

    video_to_story_free(args.video, args.out)
