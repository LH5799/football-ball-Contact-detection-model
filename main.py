from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import os
import cv2
import pandas as pd
import numpy as np
import torch
import time
import torchvision.transforms as T
from torchvision.transforms.functional import resize as tv_resize
from torchvision.transforms.functional import to_tensor, normalize
import torch.nn.functional as F

start = time.perf_counter()

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Modelle laden
object_model = YOLO("PATH_TO_PLAYER_DETECTION_MODEL").to(device)
pose_model = YOLO("yolo11x-pose.pt").to(device)

#Modell zur Tiefenschätzung
midas_model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type).to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if midas_model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

#Tracker initialisieren
tracker = sv.ByteTrack()

# Video-Pfade
video_path = "PATH_TO_VIDEO"
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
    results = object_model(image, conf=0.55)[0]
    sv_detections = sv.Detections.from_ultralytics(results)
    tracked_detections = tracker.update_with_detections(sv_detections)

    player_crops = []
    player_boxes = []
    tracked_ids = []
    frame_keypoints = []

    used_ids = set()

    ball_contact_detected = False 

    for i in range(len(tracked_detections)):
        box = tracked_detections.xyxy[i]
        cls_id = int(tracked_detections.class_id[i])
        conf = float(tracked_detections.confidence[i])
        class_name = object_model.names[cls_id]

        if class_name == "ball":
            track_id = 1
        else:
            original_id = tracked_detections.tracker_id[i]
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

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, f"{class_name} {conf:.2f} ID:{track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
        frame_keypoints = []


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

                frame_keypoints.append([img_name, tracked_ids[i]] + keypoints_flat)

    # Hole Ball-Position (Mittelpunkt)
    ball_coords = None
    x1b = y1b = x2b = y2b = None 
    for i in range(len(tracked_detections)):
        ball_class_id = next(k for k, v in object_model.names.items() if v == "ball")
        if int(tracked_detections.class_id[i]) == ball_class_id:
            x1b, y1b, x2b, y2b = map(int, tracked_detections.xyxy[i])
            ball_coords = ((x1b + x2b) // 2, (y1b + y2b) // 2)
            break
    
    # Nur wenn Ball erkannt wurde
    if ball_coords and x1b is not None:
        bx, by = ball_coords


        close_keypoints = []

        for row in frame_keypoints:
            person_id = row[1]
            for kp_idx in range(len(keypoint_names)):
                x = row[2 + kp_idx * 2]
                y = row[2 + kp_idx * 2 + 1]

                if not np.isnan(x) and not np.isnan(y):
                    px, py = x, y

                    # Berechne Abstand des Keypoints zur Bounding Box
                    dx = max(x1b - px, 0, px - x2b)
                    dy = max(y1b - py, 0, py - y2b)
                    dist_to_box = np.hypot(dx, dy)


                    #Berechne Ballgröße
                    ball_width = x2b - x1b
                    ball_height = y2b - y1b
                    ball_diag = np.sqrt(ball_width ** 2 + ball_height ** 2)

                    # Setze Schwelle in Relation zur Ballgröße
                    dynamic_threshold = max(ball_diag * 0.4, 30)

                    #if dist_2d < dynamic_threshold:
                    if dist_to_box < dynamic_threshold:
                        close_keypoints.append((int(x), int(y), person_id, keypoint_names[kp_idx]))

        # Wenn mindestens ein Keypoint nah: MiDaS Tiefe berechnen
        if close_keypoints:
            # MiDaS-Eingabe vorbereiten
            input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_transformed = transform(input_rgb)
            if isinstance(input_transformed, tuple):
                input_transformed = input_transformed[0]
            input_transformed = input_transformed.to(device)

            with torch.no_grad():
                prediction = midas(input_transformed)
                prediction_cpu = prediction.cpu()

                prediction_resized = torch.nn.functional.interpolate(
                    prediction_cpu.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()

                # Tiefe des Balls bestimmen
                ball_depth = prediction_resized[by, bx].item()

                for (x, y, person_id, kp_name) in close_keypoints:
                    keypoint_depth = prediction_resized[int(y), int(x)].item()
                    depth_diff = abs(keypoint_depth - ball_depth)

                    # Schwelle für "ähnliche Tiefe" (z. B. 10% des Tiefenwerts oder absolut < 0.1)
                    relative_thresh = 0.03 * ball_depth
                    absolute_thresh = 0.005
                    if depth_diff < max(relative_thresh, absolute_thresh):
                        ball_contact_detected = True
                        cv2.circle(image, (x, y), 7, (0, 255, 255), 2)
                        cv2.putText(image, f"Kontakt: {kp_name}", (x + 5, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
    # Anzeige der Ballkontakt-Information
    label_text = "potenzieller Ballkontakt" if ball_contact_detected else "kein Ballkontakt"

    # Schrift-Eigenschaften
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    font_thickness = 3
    text_size, _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)

    # Position: zentriert unten
    text_x = (width - text_size[0]) // 2
    text_y = height - 30  # 30 Pixel vom unteren Rand

    # Hintergrund-Rechteck
    rect_x1 = text_x - 20
    rect_y1 = text_y - text_size[1] - 20
    rect_x2 = text_x + text_size[0] + 20
    rect_y2 = text_y + 20

    # Zeichne schwarzen Hintergrund
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), thickness=-1)

    # Zeichne Text
    text_color = (255, 255, 255) if not ball_contact_detected else (255, 0, 255)
    cv2.putText(image, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    cv2.imwrite(output_img_path, image)
    out_video.write(image)
    frame_idx += 1

# Cleanup
cap.release()
out_video.release()

print("✅ Optimiert fertig – Einzel-Inferenz verwendet")

end = time.perf_counter()
print(f"⏱️ Verarbeitungszeit: {end - start:.2f} Sekunden")