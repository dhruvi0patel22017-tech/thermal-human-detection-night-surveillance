"""
GHOSTTRACK: Thermal Signature-Based Human Detection System
M.Sc. (AI-ML) - Image Processing Mini Project

USAGE:
    python ghosttrack.py --input <path> [--fps 15] [--output result.avi]

    --input can be EITHER:
        1) A folder of images (e.g. thermal frames: frame001.jpg, frame002.jpg, ...)
        2) A single video file (e.g. thermal_video.mp4 / .avi)

If a folder is given, the script first stitches the images into a video
(images_to_video), then runs the GhostTrack detection pipeline on it.
If a video file is given, it skips straight to detection.

OUTPUT:
    - An annotated output video with detection boxes drawn
    - A CSV log of every detection (frame number, position, confidence)
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import glob
import argparse
import time
from datetime import datetime


# ============================================
# STEP 1: IMAGE FOLDER -> VIDEO CONVERTER
# ============================================

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')


def images_to_video(folder_path, output_path, fps=15):
    """
    Reads all images in a folder (sorted by filename) and stitches
    them into a single video file.
    """
    image_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not image_files:
        raise ValueError(f"No images found in folder: {folder_path}")

    print(f"Found {len(image_files)} images in '{folder_path}'.")

    # Read the first image to get frame size
    first_frame = cv2.imread(os.path.join(folder_path, image_files[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, filename)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"  Skipping unreadable file: {filename}")
            continue

        # Resize in case some images differ in size from the first one
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)

        if i % 20 == 0 or i == len(image_files):
            print(f"  Converted {i}/{len(image_files)} frames...")

    writer.release()
    print(f"Video created: {output_path} ({len(image_files)} frames @ {fps} fps)\n")


def is_video_file(path):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
    return os.path.isfile(path) and path.lower().endswith(video_extensions)


def is_image_folder(path):
    if not os.path.isdir(path):
        return False
    return any(f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(path))


# ============================================
# STEP 2: GHOSTTRACK DETECTION PIPELINE
# ============================================

class GhostTrack:
    """
    Thermal human-detection pipeline using HOG (Histogram of Oriented
    Gradients) + a linear SVM person detector (OpenCV's built-in
    pretrained pedestrian detector), on top of contrast-enhanced
    thermal frames. This is far more reliable than plain brightness/
    shape thresholding, because it actually matches the gradient
    pattern of a human silhouette instead of just "bright blob shape".

    Pipeline per frame:
        1. Enhance (histogram equalization to bring out heat contrast)
        2. Upscale 2x (HOG's default 64x128 window needs a reasonably
           sized subject; thermal frames are often small/low-res)
        3. Run HOG people detector -> raw boxes + confidence weights
        4. Filter out low-confidence boxes (removes false positives
           like poles, reflections, awnings)
        5. Non-Max Suppression (removes duplicate overlapping boxes)
    """

    # Detections with weight below this are discarded as likely false
    # positives (poles, reflections, structural edges). Raise this if
    # you still see false positives; lower it if real (often distant/
    # small) people are being missed.
    WEIGHT_THRESHOLD = 0.8
    NMS_THRESHOLD = 0.3
    # Thermal frames are often small/low-res, and distant pedestrians
    # can be very small in pixel terms. Upscaling more (4x) lets HOG's
    # fixed-size detection window pick up smaller/farther people, at
    # the cost of a few more false positives on bright edges.
    UPSCALE_FACTOR = 4

    def __init__(self):
        self.detection_log = []
        self.frame_count = 0
        self.human_count = 0
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def enhance(self, gray):
        # Contrast stretching
        min_val, max_val = np.min(gray), np.max(gray)
        if max_val > min_val:
            gray = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        # Histogram equalization brings out the heat signature contrast
        gray = cv2.equalizeHist(gray)
        return gray

    def detect_people(self, gray):
        """Runs HOG detection and returns filtered (box, weight) pairs."""
        upscaled = cv2.resize(gray, None, fx=self.UPSCALE_FACTOR, fy=self.UPSCALE_FACTOR)
        upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

        boxes, weights = self.hog.detectMultiScale(
            upscaled_bgr, winStride=(4, 4), padding=(16, 16), scale=1.03
        )

        if len(boxes) == 0:
            return []

        weights = weights.flatten()

        # Drop low-confidence detections first
        keep_mask = weights > self.WEIGHT_THRESHOLD
        boxes = boxes[keep_mask]
        weights = weights[keep_mask]

        if len(boxes) == 0:
            return []

        # Non-Max Suppression to remove duplicate/overlapping boxes
        indices = cv2.dnn.NMSBoxes(
            [[int(x), int(y), int(w), int(h)] for x, y, w, h in boxes],
            weights.tolist(),
            score_threshold=self.WEIGHT_THRESHOLD,
            nms_threshold=self.NMS_THRESHOLD,
        )
        indices = np.array(indices).flatten() if len(indices) > 0 else []

        results = []
        for i in indices:
            x, y, w, h = boxes[i]
            # Scale coordinates back down since we upscaled the frame
            x, y, w, h = x // self.UPSCALE_FACTOR, y // self.UPSCALE_FACTOR, \
                         w // self.UPSCALE_FACTOR, h // self.UPSCALE_FACTOR
            results.append(((x, y, w, h), float(weights[i])))

        return results

    def process_frame(self, frame):
        self.frame_count += 1
        annotated = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        enhanced = self.enhance(gray)

        detections = []
        for (x, y, w, h), weight in self.detect_people(enhanced):
            self.human_count += 1

            # Normalize the raw SVM weight to a rough 0-1 display confidence
            display_conf = min(0.99, weight / 4.0)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(annotated, (x, y - 22), (x + w, y), (0, 0, 255), -1)
            cv2.putText(annotated, f"Human {display_conf*100:.0f}%", (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            detections.append({
                'frame': self.frame_count,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'x': x, 'y': y, 'w': w, 'h': h,
                'confidence': round(display_conf, 2)
            })

        self.detection_log.extend(detections)
        return annotated, detections

    def process_video(self, input_path, output_path, log_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {input_path}")
        print(f"Total frames: {total_frames}, FPS: {fps:.1f}")
        print("-" * 50)

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _ = self.process_frame(frame)

            info = f"Frame: {self.frame_count}/{total_frames} | Detections: {self.human_count}"
            cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            writer.write(annotated)

            if self.frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"  {self.frame_count}/{total_frames} frames | "
                      f"{self.human_count} detections so far | {elapsed:.1f}s elapsed")

        cap.release()
        writer.release()

        elapsed = time.time() - start_time
        print("-" * 50)
        print(f"Done. Frames: {self.frame_count} | Detections: {self.human_count} | Time: {elapsed:.1f}s")

        if self.detection_log:
            pd.DataFrame(self.detection_log).to_csv(log_path, index=False)
            print(f"Detection log saved: {log_path}")
        else:
            print("No detections logged (empty CSV not created).")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="GhostTrack: thermal human detection")
    parser.add_argument('--input', required=True, help='Path to an image folder OR a video file')
    parser.add_argument('--fps', type=int, default=15, help='FPS to use when converting images to video')
    parser.add_argument('--output', default='ghosttrack_output.avi', help='Path for annotated output video')
    parser.add_argument('--log', default='detection_log.csv', help='Path for the detection CSV log')
    args = parser.parse_args()

    input_path = args.input

    if is_image_folder(input_path):
        print("Input is an image folder -> converting to video first...\n")
        stitched_video = os.path.join(os.path.dirname(args.output) or '.', 'stitched_input.avi')
        images_to_video(input_path, stitched_video, fps=args.fps)
        video_to_process = stitched_video

    elif is_video_file(input_path):
        print("Input is already a video file -> skipping conversion.\n")
        video_to_process = input_path

    else:
        print(f"Error: '{input_path}' is not a valid image folder or video file.")
        sys.exit(1)

    tracker = GhostTrack()
    tracker.process_video(video_to_process, args.output, args.log)


if __name__ == "__main__":
    main()
