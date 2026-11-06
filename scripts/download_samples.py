"""
Sample Downloader & Synthetic Baxter Generator.
Ensures reproducible data availability for Baxter robot representation learning experiments,
matching Phung_Nhat_Minh_BT_GK.ipynb.
"""

from pathlib import Path
import urllib.request
import os
import cv2
import numpy as np
from PIL import Image, ImageOps


BAXTER_VIDEO_URL = "https://upload.wikimedia.org/wikipedia/commons/4/4a/Baxter_full.webm"


def generate_synthetic_baxter(video_path: Path, image_path: Path, num_frames: int = 75, size: tuple = (256, 256)):
    """Generates synthetic Baxter robot manipulation motion if offline/network fails."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 25.0, size)

    for t in range(num_frames):
        f = np.ones((size[1], size[0], 3), dtype=np.uint8) * 235
        # Workbench and background
        cv2.rectangle(f, (0, 170), (256, 256), (205, 210, 215), -1)
        cv2.rectangle(f, (100, 180), (250, 240), (70, 75, 80), -1)
        cv2.rectangle(f, (30, 90), (110, 230), (45, 50, 180), -1)
        cv2.circle(f, (70, 65), 25, (220, 220, 220), -1)
        
        # Kinematic arm motion
        ang = np.sin(t * 0.08) * 0.35
        elbow = (int(140 + np.sin(ang) * 15), int(128 + np.cos(ang) * 10))
        wrist = (int(185 + np.cos(ang) * 20), int(175 - np.sin(ang) * 15))
        cv2.line(f, (95, 120), elbow, (40, 45, 190), 14)
        cv2.line(f, elbow, wrist, (60, 65, 70), 10)
        cv2.circle(f, elbow, 9, (30, 30, 30), -1)
        out.write(f)
        if t == num_frames // 2:
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).save(image_path, quality=95)

    out.release()
    print(f"Generated synthetic Baxter motion video at: {video_path}")
    print(f"Saved representative frame at: {image_path}")


def download_or_create_samples(project_root: Path):
    data_raw = project_root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)

    vid_target = data_raw / "sample_video.mp4"
    img_target = data_raw / "sample_image.jpg"

    if vid_target.exists() and img_target.exists():
        print(f"Data samples already exist and are ready at {data_raw}")
        return

    print("Downloading Baxter robot video from Wikimedia Commons...")
    temp_webm = data_raw / "temp_baxter.webm"
    req = urllib.request.Request(BAXTER_VIDEO_URL, headers={'User-Agent': 'WorldModelResearch/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r, open(temp_webm, "wb") as f:
            f.write(r.read())

        # Extract 75 continuous manipulation frames (frames 300 to 375)
        cap = cv2.VideoCapture(str(temp_webm))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(vid_target), fourcc, 25.0, (256, 256))

        frames = []
        for _ in range(75):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = ImageOps.fit(Image.fromarray(rgb), (256, 256), method=Image.Resampling.LANCZOS)
            out.write(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            frames.append(pil_img)
        out.release()
        cap.release()

        if temp_webm.exists():
            os.remove(temp_webm)

        # Save representative high-fidelity frame (frame 337)
        if len(frames) > 0:
            frames[len(frames) // 2].save(img_target, quality=95)
            print(f"-> Saved Baxter video: {vid_target}")
            print(f"-> Saved Baxter image: {img_target}")
        else:
            raise ValueError("No frames extracted from downloaded webm.")

    except Exception as e:
        print(f"Download encounter an issue ({e}). Initializing synthetic Baxter generator...")
        generate_synthetic_baxter(vid_target, img_target)


if __name__ == "__main__":
    current_root = Path(__file__).resolve().parent.parent
    download_or_create_samples(current_root)
