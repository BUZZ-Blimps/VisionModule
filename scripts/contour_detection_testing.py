import cv2
import numpy as np

# === CONFIGURATION ===
# video_path = "Videos/yellow_circle.avi"
# video_path = "Videos/yellow_square.avi"
# video_path = "Videos/orange_circle.avi"
video_path = "Videos/orange_square.avi"

# Initial HSV min/max values
orange_hsv = {
    "h_min": 0,
    "h_max": 37,
    "s_min": 124,
    "s_max": 255,
    "v_min": 240,
    "v_max": 255
}

yellow_hsv = {
    "h_min": 27,
    "h_max": 37,
    "s_min": 107,
    "s_max": 255,
    "v_min": 197,
    "v_max": 255
}

initial_hsv = orange_hsv
# initial_hsv = yellow_hsv

def has_good_area_and_aspect(contour, min_area=500, max_aspect_ratio=1.5):
    area = cv2.contourArea(contour)
    if area < min_area:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w / h, h / w)
    return aspect_ratio <= max_aspect_ratio

def approximate_polygon(contour, N):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == N:
        return approx

    hull = cv2.convexHull(contour)
    if len(hull) < N:
        return None

    indices = np.linspace(0, len(hull)-1, N, dtype=int)
    return hull[indices]

def get_centroid(polygon):
    M = cv2.moments(polygon)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def scale_polygon(polygon, center, scale):
    cx, cy = center
    scaled = []
    for pt in polygon:
        x, y = pt[0]
        x_scaled = int(cx + scale * (x - cx))
        y_scaled = int(cy + scale * (y - cy))
        scaled.append([[x_scaled, y_scaled]])
    return np.array(scaled, dtype=np.int32)

def score_polygon_regions(mask, outer_polygon, inner_polygon):
    ring_mask = np.zeros_like(mask)
    inner_mask = np.zeros_like(mask)

    cv2.fillPoly(ring_mask, [outer_polygon], 255)
    cv2.fillPoly(inner_mask, [inner_polygon], 255)

    ring_only = cv2.subtract(ring_mask, inner_mask)

    ring_values = cv2.bitwise_and(mask, mask, mask=ring_only)
    inner_values = cv2.bitwise_and(mask, mask, mask=inner_mask)

    ring_score = np.count_nonzero(ring_values) / (np.count_nonzero(ring_only) + 1e-5)
    inner_penalty = np.count_nonzero(inner_values) / (np.count_nonzero(inner_mask) + 1e-5)

    return max(0.0, ring_score - inner_penalty)

def evaluate_contour_quality(mask, contour, N, min_area=500, max_aspect_ratio=1.5):
    if not has_good_area_and_aspect(contour, min_area, max_aspect_ratio):
        return 0.0

    polygon = approximate_polygon(contour, N)
    if polygon is None or len(polygon) != N:
        return 0.0

    centroid = get_centroid(polygon)
    if centroid is None:
        return 0.0

    inner_polygon = scale_polygon(polygon, centroid, 0.6)
    score = score_polygon_regions(mask, polygon, inner_polygon)

    return score

def fit_circle_and_score(mask, contour, scale=0.6):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    if radius < 5:
        return 0.0  # Skip very small circles

    # Outer and inner masks
    outer_mask = np.zeros_like(mask)
    inner_mask = np.zeros_like(mask)

    cv2.circle(outer_mask, center, radius, 255, thickness=-1)
    cv2.circle(inner_mask, center, int(radius * scale), 255, thickness=-1)

    ring_only = cv2.subtract(outer_mask, inner_mask)

    ring_values = cv2.bitwise_and(mask, mask, mask=ring_only)
    inner_values = cv2.bitwise_and(mask, mask, mask=inner_mask)

    ring_score = np.count_nonzero(ring_values) / (np.count_nonzero(ring_only) + 1e-5)
    inner_penalty = np.count_nonzero(inner_values) / (np.count_nonzero(inner_mask) + 1e-5)

    return max(0.0, ring_score - inner_penalty)

def evaluate_contour_circle_quality(mask, contour, min_area=500, max_aspect_ratio=1.5):
    if not has_good_area_and_aspect(contour, min_area, max_aspect_ratio):
        return 0.0

    return fit_circle_and_score(mask, contour)

def test_and_draw_contour_scores(image, mask, N=6, score_threshold=0.2):
    output = image.copy()

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return

    hierarchy = hierarchy[0]  # Get the actual hierarchy array

    for i in range(len(contours)):
        if hierarchy[i][3] != -1:
            continue  # Skip child contours, only process parents

        contour = contours[i]
        score = evaluate_contour_quality(mask, contour, N)

        if score < score_threshold:
            continue

        polygon = approximate_polygon(contour, N)
        if polygon is None or len(polygon) != N:
            continue

        cv2.polylines(output, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        centroid = get_centroid(polygon)
        if centroid:
            cx, cy = centroid
            cv2.putText(output, f"{score:.2f}", (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Scored Contours", output)

def draw_bounding_box(image, contour, color=(0, 0, 255), thickness=3):
    x, y, w, h = cv2.boundingRect(contour)
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    print("Bounding box area:", w*h)


# ======================

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
paused = False
frame = None

# Trackbar callback
def nothing(x):
    pass

# Create control window
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)

# Create HSV trackbars with initial values
cv2.createTrackbar("H Min", "Controls", initial_hsv["h_min"], 179, nothing)
cv2.createTrackbar("H Max", "Controls", initial_hsv["h_max"], 179, nothing)
cv2.createTrackbar("S Min", "Controls", initial_hsv["s_min"], 255, nothing)
cv2.createTrackbar("S Max", "Controls", initial_hsv["s_max"], 255, nothing)
cv2.createTrackbar("V Min", "Controls", initial_hsv["v_min"], 255, nothing)
cv2.createTrackbar("V Max", "Controls", initial_hsv["v_max"], 255, nothing)

while True:
    # Always read the current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video.")
        break

    # Convert to HSV
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV bounds from sliders
    h_min = cv2.getTrackbarPos("H Min", "Controls")
    h_max = cv2.getTrackbarPos("H Max", "Controls")
    s_min = cv2.getTrackbarPos("S Min", "Controls")
    s_max = cv2.getTrackbarPos("S Max", "Controls")
    v_min = cv2.getTrackbarPos("V Min", "Controls")
    v_max = cv2.getTrackbarPos("V Max", "Controls")

    lower_bound = (h_min, s_min, v_min)
    upper_bound = (h_max, s_max, v_max)

    # Create mask and result
    mask = cv2.inRange(image_HSV, lower_bound, upper_bound)
    masked_result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Dilate mask
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=3)
    cv2.imshow('Mask Dilated',dilated)

    score_threshold = 0.2
    N = 4

    # test_and_draw_contour_scores(frame, mask, 4, 0.2)

    # Find contours
    image_contours = frame.copy()
    image_bounding_boxes = frame.copy()
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    # print("Num contours:", num_contours)
    contour_areas = np.zeros((num_contours,1))
    for i in range(num_contours):
        contour_areas[i] = cv2.contourArea(contours[i])
    sorted_indices = np.argsort(contour_areas, axis=0)[::-1]
    # print(sorted_indices)

    for i in range(num_contours):
        contour_index = int(sorted_indices[i])
        contour = contours[contour_index]
        contour_score_poly = evaluate_contour_quality(mask, contour, N)
        contour_score_circle = evaluate_contour_circle_quality(mask, contour)
        contour_score = max(contour_score_poly, contour_score_circle)

        if(contour_score >= score_threshold):
            
            cv2.drawContours(image_contours, contours, contour_index, (0,0,255), 5)
            draw_bounding_box(image_bounding_boxes, contour)

            polygon = approximate_polygon(contour, N)
            if polygon is None or len(polygon) != N:
                continue

            # cv2.polylines(image_contours, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            centroid = get_centroid(polygon)
            if centroid:
                cx, cy = centroid
                cv2.putText(image_contours, f"{contour_score:.2f}", (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

    
    # print("Unsorted:", contour_areas)

    # print("Areas:")
    # for i in range(num_contours):
    #     print("\t", sorted_indices[i], ": ", contour_areas[sorted_indices[i]], sep='')
    # print("contours:",len(contours))
    # print("largest contour has ",len(contours[0]),"points")

    # largest_contour_index = int(sorted_indices[0])
    # print("Largest contour index:", largest_contour_index)

    # cv2.drawContours(image_contours, contours, largest_contour_index, (0,0,255), 5)
    cv2.imshow("Contours", image_contours)
    cv2.imshow("Bounding Boxes", image_bounding_boxes)


    # Create true hue image
    true_hue = image_HSV.copy()
    true_hue[:, :, 1] = 255
    true_hue[:, :, 2] = 255
    true_hue_bgr = cv2.cvtColor(true_hue, cv2.COLOR_HSV2BGR)

    # Display all side-by-side
    combined = cv2.hconcat([frame, masked_result, true_hue_bgr])
    cv2.imshow("Original | Masked | True Hue", combined)

    # Key input
    scrubbing_speed = 3
    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == 81:  # Left arrow
        current_frame = max(0, current_frame - scrubbing_speed)
        paused = True
    elif key == 83:  # Right arrow
        current_frame = min(total_frames - 1, current_frame + scrubbing_speed)
        paused = True
    elif not paused:
        current_frame = min(current_frame + 1, total_frames - 1)

cap.release()
cv2.destroyAllWindows()
