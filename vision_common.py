"""
vision_common.py
Shared vision utilities for the INA700 pick-and-place project.

IMPORTANT: import this from BOTH the Kaggle training notebook and
main_vision.py. Feature extraction (and now marker handling) must never
fork into two copies again - that mismatch was the root cause of the
accuracy problem.
"""

import cv2
import numpy as np

# Bump this whenever you edit this file, and print it in the notebook
# right after importing - the fastest way to confirm Kaggle is actually
# using the version you think it is, instead of a stale cached copy.
VISION_COMMON_VERSION = "v4_auto_polarity_mask"

# ---------------------------------------------------------------------------
# ArUco marker configuration
# ---------------------------------------------------------------------------

# Confirmed against capture_5_48.jpg by brute-force testing every predefined
# OpenCV dictionary - this one detected id=46 correctly. If you swap markers
# later, re-run that same brute-force check before assuming the dictionary.
ARUCO_DICT = cv2.aruco.DICT_ARUCO_MIP_36H12

# TODO: CONFIRM THIS NUMBER. Set to the marker's real printed side length in
# millimeters (you mentioned "5x5" - if that means 5 cm, this should be 50.0).
# Every distance calculation below is only as correct as this value.
MARKER_SIZE_MM = 50.0

_aruco_params = cv2.aruco.DetectorParameters()
_aruco_params.adaptiveThreshWinSizeMin = 3
_aruco_params.adaptiveThreshWinSizeMax = 53
_aruco_params.adaptiveThreshWinSizeStep = 4
_aruco_params.minMarkerPerimeterRate = 0.01
_aruco_params.maxMarkerPerimeterRate = 4.0
_aruco_params.polygonalApproxAccuracyRate = 0.05
_aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
_aruco_detector = cv2.aruco.ArucoDetector(_aruco_dictionary, _aruco_params)


def detect_marker(img_bgr):
    """
    Detect the ArUco marker in a BGR image.

    Returns None if no marker is found, otherwise a dict with:
        corners    - (4,2) float32 array of marker corners in pixel coords
        center_px  - (cx, cy) marker center in pixels
        mm_per_px  - scale factor computed from the marker's known real size
        angle_deg  - marker's rotation in the image plane (degrees)
        bbox       - (x, y, w, h) padded axis-aligned box, for masking it out
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    corners, ids, _ = _aruco_detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    c = corners[0][0]  # first detected marker: 4 points, TL/TR/BR/BL order
    side_lengths = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    avg_side_px = float(np.mean(side_lengths))
    mm_per_px = MARKER_SIZE_MM / avg_side_px

    center_px = c.mean(axis=0)

    # Angle of the TL->TR edge = marker's rotation in the image plane.
    # Used to rotate shape offsets into the marker's (robot-aligned) frame.
    edge = c[1] - c[0]
    angle_deg = float(np.degrees(np.arctan2(edge[1], edge[0])))

    x, y, w, h = cv2.boundingRect(c.astype(np.int32))
    pad = int(0.15 * max(w, h))  # avoid leaving a sliver of the marker unmasked
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = w + 2 * pad, h + 2 * pad

    return {
        "corners": c,
        "center_px": (float(center_px[0]), float(center_px[1])),
        "mm_per_px": mm_per_px,
        "angle_deg": angle_deg,
        "bbox": (x, y, w, h),
    }


def pixel_to_robot_offset(shape_px, marker_info):
    """
    Convert a shape's pixel position into a rotation-corrected (dx_mm, dy_mm)
    offset from the marker's center.

    Usage: add this offset to the marker's known robot-frame XY (measured
    once by jogging the robot to the marker's physical center and reading
    its position) to get the shape's robot-frame pick position:

        marker_robot_x, marker_robot_y = 300.0, 150.0   # measured once
        dx_mm, dy_mm = pixel_to_robot_offset((cx, cy), marker_info)
        pick_x = marker_robot_x + dx_mm
        pick_y = marker_robot_y + dy_mm
    """
    if marker_info is None:
        raise ValueError("No marker detected in this frame - cannot compute robot offset.")

    mx, my = marker_info["center_px"]
    dx_px = shape_px[0] - mx
    dy_px = shape_px[1] - my

    # Rotate the pixel offset by -angle so it lines up with the marker's
    # (and therefore the robot's calibrated) axes, then scale to mm.
    theta = -np.radians(marker_info["angle_deg"])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx_rot = dx_px * cos_t - dy_px * sin_t
    dy_rot = dx_px * sin_t + dy_px * cos_t

    mm_per_px = marker_info["mm_per_px"]
    return dx_rot * mm_per_px, dy_rot * mm_per_px


# ---------------------------------------------------------------------------
# Shared shape feature extraction (color-free) - ONE definition, used by
# both the training notebook and main_vision.py. Do not fork this again.
# ---------------------------------------------------------------------------

# How many degrees each shape repeats after - used to pick the smallest
# rotation correction needed instead of a full 0-360 range.
SHAPE_SYMMETRY_DEGREES = {
    "Circle": None,    # no meaningful orientation
    "Square": 90.0,
    "Hexagon": 60.0,
    "Star": 72.0,      # 5-pointed star
    "Triangle": 120.0,
}


def _shape_mask_from_bgr(img_bgr, exclude_bbox=None):
    blurred = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Otsu's threshold picks a cutoff, but WHICH side is "the shape" depends
    # on lighting - under some conditions THRESH_BINARY_INV masks the shape,
    # under others (e.g. different exposure) it masks almost the entire
    # frame instead (background reads as the "foreground" side of the
    # cutoff). Rather than assume one fixed polarity, compute both and keep
    # whichever produces the SMALLER foreground area - a shape sitting on a
    # table is always a small minority of the frame, so the smaller-area
    # mask is reliably the correct one regardless of lighting direction.
    _, mask_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask_inv if cv2.countNonZero(mask_inv) <= cv2.countNonZero(mask_norm) else mask_norm

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if exclude_bbox is not None:
        x, y, w, h = exclude_bbox
        mask[y:y + h, x:x + w] = 0
    return mask


def find_valid_contours(img_bgr, exclude_bbox=None):
    """
    Run the shared (color-free) mask + contour search and return every
    contour in the valid area range - used by main_vision.py, which needs
    to classify *all* shapes visible in a frame, not just the largest.

    exclude_bbox: optional (x, y, w, h) region to blank out before contour
                  search - pass the ArUco marker's bbox here (from
                  detect_marker()) so the marker is never mistaken for a shape.
    """
    mask = _shape_mask_from_bgr(img_bgr, exclude_bbox)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if 4000 < cv2.contourArea(c) < 80000]


def compute_contour_features(contour):
    """
    Given a single contour, compute the shared 13-element feature vector:
        [circularity, solidity, extent, num_vertices, aspect_ratio,
         num_convexity_defects] + 7 Hu moments
    NO COLOR - this must match on both the training and inference side.
    Returns None if the contour is degenerate (zero area/perimeter).
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0 or area == 0:
        return None

    x, y, bw, bh = cv2.boundingRect(contour)
    aspect_ratio = float(bw) / bh if bh > 0 else 0.0
    extent = float(area) / (bw * bh) if (bw * bh) > 0 else 0.0

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    num_vertices = float(len(approx))

    circularity = (4.0 * np.pi * area) / (perimeter ** 2)
    hull = cv2.convexHull(contour)
    solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0.0

    # Convexity defects: counts the "notches" between a star's points.
    # A star has ~5 pronounced defects; a hexagon/square have essentially
    # none. This is a much sharper star-vs-hexagon discriminator than
    # num_vertices alone, which approxPolyDP can round off under blur/noise.
    # Threshold is scaled by perimeter so it holds up across different
    # camera distances/shape sizes, rather than a fixed pixel cutoff.
    num_convexity_defects = 0
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3:
        hull_idx = np.sort(hull_idx, axis=0)  # required: convexityDefects needs increasing indices
        defects = cv2.convexityDefects(contour, hull_idx)
        if defects is not None:
            depth_threshold = 0.03 * perimeter * 256.0  # depths are in fixed-point (real_depth * 256)
            for i in range(defects.shape[0]):
                _, _, _, depth = defects[i, 0]
                if depth > depth_threshold:
                    num_convexity_defects += 1

    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(7):
        if hu[i] != 0:
            hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(abs(hu[i]))

    return np.array(
        [circularity, solidity, extent, num_vertices, aspect_ratio, float(num_convexity_defects)]
        + list(hu)
    )


def extract_shape_features(img_bgr, exclude_bbox=None):
    """
    Single-shape convenience wrapper for the TRAINING notebook, where each
    image has exactly one shape to learn from. Returns (features, contour)
    for the largest valid contour, or (None, None) if none is found.

    For LIVE inference with possibly multiple shapes per frame, use
    find_valid_contours() + compute_contour_features() instead (see
    main_vision.py) so every shape gets classified, not just the biggest.
    """
    valid = find_valid_contours(img_bgr, exclude_bbox)
    if not valid:
        return None, None

    c = max(valid, key=cv2.contourArea)
    features = compute_contour_features(c)
    if features is None:
        return None, None
    return features, c


def shape_pick_angle(contour, shape_name):
    """
    Angle (degrees) the gripper should add to its fixed approach orientation
    to align with the detected shape, reduced modulo that shape's rotational
    symmetry so the correction is always the smallest one needed.
    Returns 0.0 for shapes with no meaningful orientation (e.g. Circle) or
    unrecognized shape names.
    """
    symmetry = SHAPE_SYMMETRY_DEGREES.get(shape_name)
    if not symmetry:
        return 0.0

    (_, _), (_, _), raw_angle = cv2.minAreaRect(contour)
    return float(raw_angle % symmetry)


def shape_center_px(contour):
    """Centroid (cx, cy) in pixels, from image moments."""
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])
