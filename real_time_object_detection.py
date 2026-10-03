"""
Real-Time Object Detection using TensorFlow & OpenCV
===================================================
Main application for live webcam object detection with automatic Windows camera backend fallback,
real-time inference, class label and confidence percentage rendering, and live FPS overlay.

Usage:
  python real_time_object_detection.py [--camera-index INDEX] [--min-score CONFIDENCE] [--save-video]
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import tensorflow as tf

# 1. Setup dynamic relative paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(PROJECT_ROOT, "models", "research")
OBJECT_DETECTION_DIR = os.path.join(RESEARCH_DIR, "object_detection")

for path in [RESEARCH_DIR, OBJECT_DETECTION_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 2. Alias tf.gfile for TF 2.x compatibility
if not hasattr(tf, "gfile"):
    tf.gfile = tf.io.gfile

try:
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
except ImportError:
    from utils import label_map_util
    from utils import visualization_utils as vis_util

MODEL_NAME = "ssd_mobilenet_v1_coco_2017_11_17"
PATH_TO_CKPT = os.path.join(PROJECT_ROOT, MODEL_NAME, "frozen_inference_graph.pb")
PATH_TO_LABELS = os.path.join(OBJECT_DETECTION_DIR, "data", "mscoco_label_map.pbtxt")
NUM_CLASSES = 90


def find_working_camera(requested_index=None):
    """
    Scans camera backends and indices to find a working webcam delivering non-black frames.
    """
    backends_to_test = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Default OpenCV"),
    ]
    indices_to_test = [requested_index] if requested_index is not None else [0, 1, 2]

    for idx in indices_to_test:
        for backend_code, backend_name in backends_to_test:
            cap = cv2.VideoCapture(idx, backend_code)
            if not cap.isOpened():
                cap.release()
                continue

            test_frame = None
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    test_frame = frame
                    break
                time.sleep(0.05)

            if test_frame is not None:
                mean_val = np.mean(test_frame)
                print(f"[INFO] Connected to Camera (Index: {idx}, Backend: {backend_name}, Mean Brightness: {mean_val:.1f})")
                if mean_val < 1.0:
                    print(f"[WARNING] Camera frame appears dark/black (mean = {mean_val:.1f}). Check camera lens shutter.")
                return cap, idx, backend_name

            cap.release()

    return None, None, None


def load_model_and_labels():
    print(f"[INFO] Loading frozen graph from: {PATH_TO_CKPT}")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    print(f"[INFO] Loading label map from: {PATH_TO_LABELS}")
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    return detection_graph, category_index


def run_main_application(camera_index=None, min_score_thresh=0.5, save_video=False, output_path="output.avi"):
    # 1. Initialize Camera
    cap, idx, backend_name = find_working_camera(camera_index)
    if cap is None:
        print("[ERROR] Could not open any working webcam. Run 'python camera_test.py' for detailed camera diagnostics.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 2. Load Model & Label Map
    detection_graph, category_index = load_model_and_labels()

    # 3. Setup Optional Video Writer
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
        print(f"[INFO] Recording live detection feed to: {output_path}")

    # 4. TensorFlow Session & Detection Loop
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
            detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")

            print("\n========================================")
            print("REAL-TIME OBJECT DETECTION STARTED")
            print("Press 'q' in the camera window to stop.")
            print("========================================\n")

            fps = 0.0
            try:
                while True:
                    start_time = time.time()
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        print("[WARNING] Frame read timeout or empty frame received. Retrying...")
                        time.sleep(0.05)
                        continue

                    # Expand frame dims to [1, H, W, 3]
                    image_np_expanded = np.expand_dims(frame, axis=0)

                    # Execute Object Detection Inference
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded},
                    )

                    # Draw Bounding Boxes, Labels, Confidence Scores
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        min_score_thresh=min_score_thresh,
                        use_normalized_coordinates=True,
                        line_thickness=4,
                    )

                    # Calculate and Overlay FPS
                    fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - start_time, 0.001))
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.1f}",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    if video_writer is not None:
                        video_writer.write(frame)

                    # Display Live Camera Feed Window
                    cv2.imshow("Real-Time Object Detection", frame)

                    # Exit condition
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] 'q' pressed. Stopping detection stream.")
                        break
            finally:
                cap.release()
                if video_writer is not None:
                    video_writer.release()
                cv2.destroyAllWindows()
                print("[INFO] Camera released and windows closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection using TensorFlow & OpenCV")
    parser.add_argument("--camera-index", type=int, default=None, help="Specific camera index (default: auto-detect 0/1/2)")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum confidence threshold (default: 0.5)")
    parser.add_argument("--save-video", action="store_true", help="Save output detection video feed")
    parser.add_argument("--output-path", type=str, default="output.avi", help="Output path for recorded video")
    args = parser.parse_args()

    run_main_application(
        camera_index=args.camera_index,
        min_score_thresh=args.min_score,
        save_video=args.save_video,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
