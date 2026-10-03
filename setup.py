"""
Setup Check Script for Real-Time Object Detection Project
=========================================================
Verifies Python environment, TensorFlow, OpenCV, NumPy, Protobuf,
Object Detection API imports, proto bindings, model graph, and inference.

Usage:
  python setup.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(PROJECT_ROOT, "models", "research")
OBJECT_DETECTION_DIR = os.path.join(RESEARCH_DIR, "object_detection")

# Setup sys.path safely
for path in [RESEARCH_DIR, OBJECT_DETECTION_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


def verify_all():
    print("========================================")
    print("STARTING PROJECT SETUP CHECK")
    print("========================================")

    # 1. Python Check
    py_ver = sys.version.split()[0]
    print(f"[CHECK 1/9] Python Version : {py_ver}")

    # 2. TensorFlow Check
    try:
        import tensorflow as tf
        print(f"[CHECK 2/9] TensorFlow     : {tf.__version__}")
        if not hasattr(tf, "gfile"):
            tf.gfile = tf.io.gfile
    except ImportError as e:
        print(f"[FAIL] TensorFlow import failed: {e}")
        return False

    # 3. OpenCV Check
    try:
        import cv2
        print(f"[CHECK 3/9] OpenCV         : {cv2.__version__}")
    except ImportError as e:
        print(f"[FAIL] OpenCV import failed: {e}")
        return False

    # 4. NumPy Check
    try:
        import numpy as np
        print(f"[CHECK 4/9] NumPy          : {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy import failed: {e}")
        return False

    # 5. Protobuf Check
    try:
        import google.protobuf
        print(f"[CHECK 5/9] Protobuf       : {google.protobuf.__version__}")
    except ImportError as e:
        print(f"[FAIL] Protobuf import failed: {e}")
        return False

    # 6. Object Detection API Imports & Proto Check
    try:
        from object_detection.protos import string_int_label_map_pb2
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as vis_util
        print("[CHECK 6/9] Object Detection API Imports: PASSED")
    except Exception as e:
        print(f"[FAIL] Object Detection API imports failed: {e}")
        return False

    # 7. Model & Label Files Check
    model_name = "ssd_mobilenet_v1_coco_2017_11_17"
    ckpt_path = os.path.join(PROJECT_ROOT, model_name, "frozen_inference_graph.pb")
    label_path = os.path.join(OBJECT_DETECTION_DIR, "data", "mscoco_label_map.pbtxt")

    if not os.path.exists(ckpt_path):
        print(f"[FAIL] Model file missing: {ckpt_path}")
        return False
    if not os.path.exists(label_path):
        print(f"[FAIL] Label map file missing: {label_path}")
        return False
    print("[CHECK 7/9] Model Checkpoint & Label Map Files: PASSED")

    # 8. Model Graph Loading Check
    try:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(ckpt_path, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        print("[CHECK 8/9] Model Graph & Label Map Parsing: PASSED")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return False

    # 9. Basic Inference Test
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
                detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
                detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
                detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
                num_detections = detection_graph.get_tensor_by_name("num_detections:0")

                image_np_expanded = np.expand_dims(dummy_frame, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded},
                )
        print("[CHECK 9/9] Basic Model Inference Pass: PASSED")
    except Exception as e:
        print(f"[FAIL] Inference execution failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = verify_all()
    if success:
        print("\n========================================")
        print("SETUP CHECK PASSED")
        print("========================================\n")
    else:
        print("\n========================================")
        print("SETUP CHECK FAILED")
        print("========================================\n")
        sys.exit(1)
