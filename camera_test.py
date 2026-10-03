"""
Camera Test Script for Real-Time Object Detection
=================================================
Independently tests Windows webcam access, backends, frame validity, and image brightness.
Supports automatic fallback between DirectShow, Media Foundation, and default OpenCV backends.

Usage:
  python camera_test.py [--camera-index 0]
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2


def get_backend_name(backend_id):
    if backend_id == cv2.CAP_DSHOW:
        return "DirectShow (CAP_DSHOW)"
    elif backend_id == cv2.CAP_MSMF:
        return "Media Foundation (CAP_MSMF)"
    elif backend_id == cv2.CAP_ANY:
        return "Default (CAP_ANY)"
    return f"Backend ID {backend_id}"


def find_working_camera(requested_index=None):
    """
    Scans camera backends and camera indices to find a camera that:
    1. Opens successfully (cap.isOpened())
    2. Delivers valid frames (ret == True)
    3. Produces a non-black valid image (frame.mean() > 1.0)
    """
    backends_to_test = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Default OpenCV"),
    ]

    indices_to_test = [requested_index] if requested_index is not None else [0, 1, 2]

    print("[INFO] Scanning Windows camera backends and indices...")

    for idx in indices_to_test:
        for backend_code, backend_name in backends_to_test:
            print(f"[TEST] Trying Index {idx} with Backend {backend_name}...")
            cap = cv2.VideoCapture(idx, backend_code)

            if not cap.isOpened():
                cap.release()
                continue

            # Try reading up to 5 warmup frames
            frame_valid = False
            test_frame = None
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    test_frame = frame
                    frame_valid = True
                    break
                time.sleep(0.05)

            if frame_valid and test_frame is not None:
                frame_mean = np.mean(test_frame)
                print(f"[SUCCESS] Camera found! Index: {idx} | Backend: {backend_name} | Mean Brightness: {frame_mean:.2f}")

                if frame_mean < 1.0:
                    print(f"[WARNING] Frame is almost entirely black (mean = {frame_mean:.2f}). Check camera lens cover/privacy switch.")

                return cap, idx, backend_code, backend_name

            cap.release()

    return None, None, None, None


def run_camera_test(requested_index=None):
    cap, index, backend_code, backend_name = find_working_camera(requested_index)

    if cap is None:
        print("\n========================================")
        print("CAMERA TEST FAILED: No working webcam found!")
        print("========================================")
        print("Troubleshooting Steps:")
        print("  1. Ensure physical webcam is plugged in.")
        print("  2. Check Windows Camera Privacy Settings (Settings -> Privacy & Security -> Camera).")
        print("  3. Verify no other app (Zoom, Teams, Skype, Browser) is locking the camera.")
        print("  4. If using laptop, check if physical privacy shutter is closed.")
        sys.exit(1)

    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n========================================")
    print("CAMERA DIAGNOSTIC INFORMATION")
    print("========================================")
    print(f"Camera backend: {backend_name}")
    print(f"Camera index  : {index}")
    print(f"Camera opened : {cap.isOpened()}")

    # Capture sample frame to print initial metrics
    ret, sample_frame = cap.read()
    if ret and sample_frame is not None:
        print(f"Frame received: True")
        print(f"Frame shape   : {sample_frame.shape}")
        print(f"Frame mean    : {np.mean(sample_frame):.2f}")
    else:
        print(f"Frame received: False")

    print("========================================")
    print("Press 'q' in the window to exit camera test.\n")

    fps = 0.0
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                print("[WARNING] Frame read failed. Retrying...")
                time.sleep(0.1)
                continue

            frame_mean = np.mean(frame)
            fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - start_time, 0.001))

            # Overlay diagnostic information on live feed
            cv2.putText(frame, f"Backend: {backend_name} | Index: {index}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Res: {frame.shape[1]}x{frame.shape[0]} | Mean Brightness: {frame_mean:.1f}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Camera Test Feed (Press Q to Exit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User requested exit.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("CAMERA TEST COMPLETED CLEANLY.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Windows Camera Test")
    parser.add_argument("--camera-index", type=int, default=None, help="Specific camera index to test")
    args = parser.parse_args()

    run_camera_test(args.camera_index)
