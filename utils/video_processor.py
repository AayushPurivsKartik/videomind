# utils/video_processor.py
import os
import cv2
import shutil
from pathlib import Path

def extract_frames(
    video_path: str,
    output_dir: str = "frames",
    fps: int = 1
):
    """
    Extract frames at ~1 FPS (adjustable)
    Returns: list of frame paths, list of timestamps (seconds), video_name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy video to our videos/ folder for persistence
    video_name = Path(video_path).name
    dest_video_path = f"videos/{video_name}"
    os.makedirs("videos", exist_ok=True)
    shutil.copy2(video_path, dest_video_path)
    
    cap = cv2.VideoCapture(dest_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {dest_video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    
    frame_paths = []
    timestamps = []
    saved_count = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            timestamps.append(round(timestamp, 2))
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames → {output_dir}/")
    return frame_paths, timestamps, video_name