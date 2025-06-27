from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import os
import cv2
import pandas as pd
import numpy as np
import torch
import time

start = time.perf_counter()

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Modelle laden
object_model = YOLO("PATH_TO/player_detection_model.pt").to(device)
pose_model = YOLO("yolo11x-pose.pt").to(device)

#Tracker initialisieren
tracker = sv.ByteTrack()

# Video-Pfade
video_path = "INPUT_PATH"
output_dir = "OUTPUT_PATH"
os.makedirs(output_dir, exist_ok=True)

# Video laden
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional: VideoWriter
out_video = cv2.VideoWriter(os.path.join(output_dir, "processed_video.mp4"),
                            cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Keypoints
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
columns = ["Bildname", "PersonID"] + [f"{kp}_{axis}" for kp in keypoint_names for axis in ["x", "y"]]
data = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = frame.copy()
    height, width = image.shape[:2]
    img_name = f"frame_{frame_idx:05d}.jpg"
    output_img_path = os.path.join(output_dir, img_name)

    # Objekterkennung
    results = object_model(image, conf=0.5)[0]
    sv_detections = sv.Detections.from_ultralytics(results)
    tracked_detections = tracker.update_with_detections(sv_detections)

    player_crops = []
    player_boxes = []
    tracked_ids = []

    used_ids = set()

    for i in range(len(sv_detections)):
        box = sv_detections.xyxy[i]
        cls_id = int(sv_detections.class_id[i])
        conf = float(sv_detections.confidence[i])
        class_name = object_model.names[cls_id]

        # Setze ID
        if class_name == "ball":
            track_id = 1
        else:
            original_id = sv_detections.tracker_id[i]
            if original_id == 1 or original_id in used_ids:
                new_id = 2
                while new_id in used_ids:
                    new_id += 1
                track_id = new_id
            else:
                track_id = original_id
        used_ids.add(track_id)

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        # Bounding Box zeichnen
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, f"{class_name} {conf:.2f} ID:{track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Filtere nur Spieler (player, goalkeeper)
        if class_name not in ["player", "goalkeeper"]:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        player_crops.append(crop)
        player_boxes.append((x1, y1, x2, y2))
        tracked_ids.append(track_id)

    # Batch-Pose Estimation auf alle Spieler-Crops gleichzeitig
    if player_crops:
        target_size = (256, 256)  # anpassen an dein Modell, oft 256x256 oder 384x384

        player_crops_resized = []
        original_sizes = []  # WICHTIG: Speichere Originalgrößen, um Keypoints zurückzuskalierten

        for crop in player_crops:
            original_sizes.append((crop.shape[1], crop.shape[0]))  # width, height
            resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
            player_crops_resized.append(resized)

        batch = np.stack(player_crops_resized)
        batch = batch[..., ::-1]  # BGR -> RGB
        batch = batch.copy()      # copy, um negative Strides zu vermeiden

        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device).float() / 255.0
        pose_results = pose_model(batch_tensor, conf=0.55)

        # Keypoints relativ zum Crop auf Originalbild übertragen und zeichnen
        for i, pose_result in enumerate(pose_results):
            if not pose_result.keypoints or pose_result.keypoints.xy is None:
                continue

            keypoints_all = pose_result.keypoints.xy.cpu().numpy()
            x1, y1, x2, y2 = player_boxes[i]
            orig_w, orig_h = original_sizes[i]

            scale_x = orig_w / target_size[0]
            scale_y = orig_h / target_size[1]

            for person_id, person_keypoints in enumerate(keypoints_all):
                if person_keypoints.shape[0] != len(keypoint_names):
                    continue

                keypoints_flat = []
                for i_kp in range(len(keypoint_names)):
                    x_rel, y_rel = person_keypoints[i_kp, 0], person_keypoints[i_kp, 1]

                    if x_rel != 0 and y_rel != 0:
                        # Keypoints von 256x256 (resized Crop) auf Original-Crop-Größe skalieren
                        x_scaled = x_rel * scale_x
                        y_scaled = y_rel * scale_y

                        # Dann in Bildkoordinaten versetzen
                        x = x1 + x_scaled
                        y = y1 + y_scaled
                        keypoints_flat.extend([x, y])

                        cv2.circle(image, (int(x), int(y)), 5, (255, 0, 255), -1)
                        cv2.putText(image, keypoint_names[i_kp], (int(x) + 5, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    else:
                        keypoints_flat.extend([np.nan, np.nan])

                data.append([img_name, tracked_ids[i]] + keypoints_flat)

    cv2.imwrite(output_img_path, image)
    out_video.write(image)
    frame_idx += 1

# Cleanup
cap.release()
out_video.release()

# CSV speichern
df = pd.DataFrame(data, columns=columns)
df.to_csv(os.path.join(output_dir, "yolo_pose_data.csv"), index=False)

print("✅ Optimiert fertig – Einzel-Inferenz verwendet")

end = time.perf_counter()
print(f"⏱️ Verarbeitungszeit: {end - start:.2f} Sekunden")