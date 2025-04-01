import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import imutils
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# -------- MiDaS Depth Estimation Model -------- #
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True).eval()

midas_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# -------- MobileNetV3 Model for Enhanced Edge Detection -------- #
mobilenet = models.mobilenet_v3_small(pretrained=True).eval()

mobilenet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------- Constants and Calibration Factors -------- #
FOCAL_LENGTH_MM = 4.25
SENSOR_WIDTH_MM = 6.4
KNOWN_OBJECT_LENGTH_CM = 15.0
KNOWN_PIXEL_LENGTH = 300
CALIBRATION_FACTORS = {'small': 0.08, 'medium': 0.06, 'large': 0.04}
OPTIMAL_DISTANCE_CM = 40.0
LARGE_OBJECT_SCALING = 1.8
MEDIUM_OBJECT_SCALING = 1.05

# -------- Depth Estimation Function -------- #
def estimate_depth(image):
    image = image.convert("RGB")
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = midas_transform(Image.fromarray((image_np * 255).astype(np.uint8))).unsqueeze(0)

    with torch.no_grad():
        depth_map = midas(image_tensor).squeeze().cpu().numpy()

    depth_map = cv2.resize(depth_map, (image_np.shape[1], image_np.shape[0]))
    return (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-6)

# -------- Contour Detection for MiDaS Measurement -------- #
def get_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return sorted([c for c in contours if cv2.contourArea(c) > 2000], key=cv2.contourArea, reverse=True)

# -------- MiDaS Measurement Function -------- #
def estimate_length(image, unit='cm'):
    depth_map = estimate_depth(image)
    image = np.array(image)
    contours = get_contours(image)

    if not contours:
        return image, "No objects detected!"

    largest_contour = contours[0]
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect).astype(np.int32)

    length_px = max(rect[1])

    mask = np.zeros(depth_map.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    depths_inside_contour = depth_map[mask == 255]

    depth_at_object = np.percentile(depths_inside_contour, 50) if len(depths_inside_contour) > 0 else np.mean(depth_map)
    pixel_size_mm = (SENSOR_WIDTH_MM / image.shape[1]) * (depth_at_object / FOCAL_LENGTH_MM)
    object_length_cm = (length_px * pixel_size_mm) / 10

    calibration_factor = (KNOWN_OBJECT_LENGTH_CM / KNOWN_PIXEL_LENGTH) * length_px
    object_length_cm = (object_length_cm + calibration_factor) / 2

    if object_length_cm > 150:
        object_length_cm *= LARGE_OBJECT_SCALING
    elif object_length_cm > 100:
        object_length_cm *= MEDIUM_OBJECT_SCALING

    object_length = object_length_cm if unit == 'cm' else object_length_cm / 100
    unit_label = "cm" if unit == 'cm' else "m"

    # Draw the bounding box in green
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # Display the length ABOVE the bounding box (based on top-left corner of the box)
    x, y = box[0][0], box[0][1]
    y = y - 10 if y > 20 else y + 30  # Adjust position to prevent drawing outside image
    cv2.putText(image, f"Length: {object_length:.2f} {unit_label}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image, f"Length: {object_length:.2f} {unit_label}"

# -------- Live Measurement with Canny Edge Detection -------- #
points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))

def measure_length():
    global points

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    cv2.namedWindow("Live Measurement")
    cv2.setMouseCallback("Live Measurement", click_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = cv2.Canny(frame, 100, 200)
        cv2.imshow("Edge Detection", edges)

        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
            pixel_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
            real_length = pixel_dist * CALIBRATION_FACTORS['medium']

            # Modified code to show the length ABOVE the line
            mid_x = int((points[0][0] + points[1][0]) / 2)
            mid_y = int((points[0][1] + points[1][1]) / 2) - 20  # Move text above the line
            cv2.putText(frame, f"Length: {real_length:.2f} cm", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, "Optimal Distance: 20-100 cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "Press 'R' to Reset | 'Q' to Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Live Measurement", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            points = []
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------- Streamlit Interface -------- #
st.title("AI-Based Measurement App")

mode = st.radio("Select Mode:", ["Image Upload", "Live Measurement"])

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an object image", type=["jpg", "png", "jpeg"])
    unit_option = st.radio("Select Measurement Unit:", ["Centimeters (cm)", "Meters (m)"])
    unit = 'cm' if "cm" in unit_option else 'm'

    if uploaded_file:
        image = Image.open(uploaded_file)
        processed_image, result_text = estimate_length(image, unit)
        processed_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        st.image(processed_image, caption="Measured Object", use_column_width=True)
        st.write(result_text)

elif mode == "Live Measurement":
    if st.button("Start Live Measurement"):
        measure_length()
