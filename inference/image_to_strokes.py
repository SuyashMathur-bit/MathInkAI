import io
import cv2
import numpy as np
from PIL import Image

def extract_strokes_from_image(image_bytes, target_width=800, target_height=400):
    """
    Extract ordered handwriting strokes from an uploaded image (PNG, JPG, etc.).
    Handles:
    - Auto-rotation / format loading via PIL
    - Inversion detection (dark ink on light paper vs light ink on blackboard)
    - Adaptive Otsu thresholding
    - Morphological skeletonization / contour path tracing
    - Left-to-right temporal stroke ordering
    - Coordinate scaling to fit the smart board canvas
    """
    # 1. Load image using PIL to handle orientation & formats cleanly
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not decode image: {e}")

    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 2. Resize to reasonable working resolution preserving aspect ratio
    h, w = gray.shape
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # 3. Detect background polarity: light paper vs dark board
    # Check average corner brightness
    corner_pixels = [
        gray[0:10, 0:10],
        gray[0:10, -10:],
        gray[-10:, 0:10],
        gray[-10:, -10:]
    ]
    corner_mean = np.mean([np.mean(c) for c in corner_pixels])

    # Otsu thresholding
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if corner_mean > 127:
        # Dark ink on light background -> invert so ink is white (255)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Light ink on dark background
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean small speckles with open/close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 4. Skeletonize to extract stroke centerlines
    skel = np.zeros(cleaned.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = cleaned.copy()
    size = np.size(cleaned)

    for _ in range(30):
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break

    # 5. Extract contours from skeleton
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # If skeletonization produced too few points, fallback to binary contours
    if len(contours) == 0 or sum(len(c) for c in contours) < 10:
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    raw_strokes = []
    for c in contours:
        pts = c.squeeze()
        if len(pts.shape) != 2:
            continue
        if len(pts) < 4:
            continue

        # Subsample for smooth stroke sequences
        step = max(1, len(pts) // 35)
        sampled = pts[::step].tolist()
        if len(sampled) >= 2:
            # Convert to [float(x), float(y)]
            raw_strokes.append([[float(p[0]), float(p[1])] for p in sampled])

    if not raw_strokes:
        return []

    # 6. Sort strokes left-to-right (temporal handwriting flow)
    raw_strokes.sort(key=lambda s: min(p[0] for p in s))

    # 7. Scale and position strokes nicely on the target smart board canvas
    all_x = [p[0] for s in raw_strokes for p in s]
    all_y = [p[1] for s in raw_strokes for p in s]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    bw = max_x - min_x or 1
    bh = max_y - min_y or 1

    pad = 40
    avail_w = target_width - 2 * pad
    avail_h = target_height - 2 * pad

    scale = min(avail_w / bw, avail_h / bh)

    canvas_offset_x = (target_width - bw * scale) / 2.0 - min_x * scale
    canvas_offset_y = (target_height - bh * scale) / 2.0 - min_y * scale

    canvas_strokes = []
    for s in raw_strokes:
        canvas_s = []
        for p in s:
            cx = p[0] * scale + canvas_offset_x
            cy = p[1] * scale + canvas_offset_y
            canvas_s.append([round(cx, 1), round(cy, 1)])
        canvas_strokes.append(canvas_s)

    return canvas_strokes
