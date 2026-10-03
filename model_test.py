"""
Model Test Script for Real-Time Object Detection
================================================
Independently verifies:
1. Frozen graph checkpoint loading
2. Label map file loading & category index creation
3. Model inference execution on sample frame tensor
4. Class label formatting and confidence score generation

Usage:
  python model_test.py
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Setup dynamic relative paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(PROJECT_ROOT, "models", "research")
OBJECT_DETECTION_DIR = os.path.join(RESEARCH_DIR, "object_detection")

for path in [RESEARCH_DIR, OBJECT_DETECTION_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Alias tf.gfile for TF 2.x compatibility
if not hasattr(tf, "gfile"):
    tf.gfile = tf.io.gfile

try:
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
except ImportError:
    from utils import label_map_util
    from utils import visualization_utils as vis_util


def run_model_test():
    print("=== Running Model Test ===")
    model_name = "ssd_mobilenet_v1_coco_2017_11_17"
    ckpt_path = os.path.join(PROJECT_ROOT, model_name, "frozen_inference_graph.pb")
    label_path = os.path.join(OBJECT_DETECTION_DIR, "data", "mscoco_label_map.pbtxt")
    num_classes = 90

    # 1. Verify model file exists
    if not os.path.exists(ckpt_path):
        print(f"[FAIL] Model file missing at: {ckpt_path}")
        sys.exit(1)
    print(f"[OK] Found frozen graph: {ckpt_path}")

    # 2. Verify label map file exists
    if not os.path.exists(label_path):
        print(f"[FAIL] Label map file missing at: {label_path}")
        sys.exit(1)
    print(f"[OK] Found label map: {label_path}")

    # 3. Load detection graph into memory
    print("[INFO] Loading frozen graph into TensorFlow...")
    start_load = time.time()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(ckpt_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
    print(f"[OK] Graph loaded successfully in {time.time() - start_load:.2f} seconds.")

    # 4. Load label map & category index
    print("[INFO] Loading category index...")
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)
    print(f"[OK] Loaded {len(category_index)} category mappings (e.g. 1 -> '{category_index.get(1, {}).get('name')}')")

    # 5. Run inference test on a sample frame array
    print("[INFO] Running inference test on sample frame (640x480x3)...")
    sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
            detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")

            image_np_expanded = np.expand_dims(sample_frame, axis=0)

            t0 = time.time()
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded},
            )
            inference_time = (time.time() - t0) * 1000.0

            # Test visualization utilities
            vis_util.visualize_boxes_and_labels_on_image_array(
                sample_frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
            )

            print(f"[OK] Inference pass completed in {inference_time:.2f} ms.")
            print(f"[OK] Number of raw candidate detections: {int(num[0])}")

    print("\n========================================")
    print("MODEL TEST PASSED")
    print("========================================")


if __name__ == "__main__":
    run_model_test()
