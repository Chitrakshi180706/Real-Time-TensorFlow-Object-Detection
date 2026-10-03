"""
Setup and Environment Check Script for Real-Time Object Detection
------------------------------------------------------------------
Checks dependencies, builds protobuf definitions if needed,
and verifies model & label map file availability.
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(PROJECT_ROOT, "models", "research")
OBJECT_DETECTION_DIR = os.path.join(RESEARCH_DIR, "object_detection")

# Ensure required paths are in sys.path
for path in [RESEARCH_DIR, OBJECT_DETECTION_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

def check_dependencies():
    print("=== Checking Python Environment & Dependencies ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version.split()[0]}")

    try:
        import tensorflow as tf
        print(f"TensorFlow       : {tf.__version__}")
        if not hasattr(tf, 'gfile'):
            tf.gfile = tf.io.gfile
    except ImportError as e:
        print(f"ERROR: TensorFlow import failed: {e}")
        return False

    try:
        import cv2
        print(f"OpenCV           : {cv2.__version__}")
    except ImportError as e:
        print(f"ERROR: OpenCV import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"NumPy            : {np.__version__}")
    except ImportError as e:
        print(f"ERROR: NumPy import failed: {e}")
        return False

    try:
        import google.protobuf
        print(f"protobuf         : {google.protobuf.__version__}")
    except ImportError as e:
        print(f"ERROR: Protobuf import failed: {e}")
        return False

    return True

def compile_protos_if_needed():
    print("\n=== Checking Protobuf Compilations ===")
    protos_dir = os.path.join(OBJECT_DETECTION_DIR, "protos")
    proto_files = [f for f in os.listdir(protos_dir) if f.endswith(".proto")]
    missing_pb2 = []

    for pf in proto_files:
        pb2_name = pf.replace(".proto", "_pb2.py")
        if not os.path.exists(os.path.join(protos_dir, pb2_name)):
            missing_pb2.append(pf)

    if not missing_pb2:
        print("All proto files are already compiled (*_pb2.py present).")
        return True

    print(f"Found {len(missing_pb2)} uncompiled proto files. Attempting compilation with protoc.exe...")
    protoc_bin = os.path.join(PROJECT_ROOT, "protoc-3.4.0-win32", "bin", "protoc.exe")
    if not os.path.exists(protoc_bin):
        print(f"WARNING: protoc executable not found at {protoc_bin}.")
        return False

    cmd = [
        protoc_bin,
        f"-I={RESEARCH_DIR}",
        f"--python_out={RESEARCH_DIR}",
        os.path.join(OBJECT_DETECTION_DIR, "protos", "*.proto")
    ]
    try:
        res = subprocess.run(" ".join(cmd), shell=True, check=True, capture_output=True, text=True)
        print("Protobuf compilation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Protobuf compilation error: {e.stderr}")
        return False

def verify_model_files():
    print("\n=== Verifying Model and Label Map Files ===")
    model_name = "ssd_mobilenet_v1_coco_2017_11_17"
    ckpt_path = os.path.join(PROJECT_ROOT, model_name, "frozen_inference_graph.pb")
    label_path = os.path.join(OBJECT_DETECTION_DIR, "data", "mscoco_label_map.pbtxt")

    if os.path.exists(ckpt_path):
        print(f"Found Frozen Inference Graph: {ckpt_path}")
    else:
        print(f"ERROR: Frozen graph missing at {ckpt_path}")
        return False

    if os.path.exists(label_path):
        print(f"Found Label Map: {label_path}")
    else:
        print(f"ERROR: Label map missing at {label_path}")
        return False

    return True

def test_imports_and_loading():
    print("\n=== Testing Object Detection API Imports & Graph Loading ===")
    try:
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as vis_util
        print("Successfully imported label_map_util and visualization_utils!")
    except Exception as e:
        print(f"ERROR importing object detection utilities: {e}")
        return False

    import tensorflow as tf
    model_name = "ssd_mobilenet_v1_coco_2017_11_17"
    ckpt_path = os.path.join(PROJECT_ROOT, model_name, "frozen_inference_graph.pb")
    label_path = os.path.join(OBJECT_DETECTION_DIR, "data", "mscoco_label_map.pbtxt")

    try:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(ckpt_path, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        print("Successfully loaded model graph definition!")

        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        print("Successfully loaded label map and created category index!")
        return True
    except Exception as e:
        print(f"ERROR during model/label map loading: {e}")
        return False

if __name__ == "__main__":
    ok = check_dependencies() and compile_protos_if_needed() and verify_model_files() and test_imports_and_loading()
    if ok:
        print("\n>>> SETUP CHECK PASSED: Environment and model setup are fully ready! <<<")
    else:
        print("\n>>> SETUP CHECK FAILED: Please review the error messages above. <<<")
        sys.exit(1)
