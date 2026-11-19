"""Decode raw video frames independently of engine sampling and image policies."""

import cv2
import numpy as np


def decode_bgr_frame(mp4: str, index: int) -> np.ndarray:
    """Read one frame by decode index, releasing the video on success or failure.

    Args:
        mp4: Video file path.
        index: Zero-based frame index.

    Returns:
        The decoded BGR image without resizing or encoding.

    Raises:
        FileNotFoundError: The video cannot be opened.
        RuntimeError: The requested frame cannot be decoded.
    """
    capture = cv2.VideoCapture(mp4)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {mp4}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {index} from {mp4}")
    finally:
        capture.release()
    return image
