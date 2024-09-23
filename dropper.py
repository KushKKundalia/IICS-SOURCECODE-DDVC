import cv2
import mediapipe as mp
from pynput.mouse import Controller as MouseController, Button

# Initialize MediaPipe hands and mouse controller
mp_hands = mp.solutions.hands
mouse = MouseController()
hands = mp_hands.Hands(min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

# Set video capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables
prev_mouse_pos = None
is_dragging = False

# Function to process video frames
def process_frame(frame):
    global prev_mouse_pos, is_dragging

    # Flip and convert the frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmark_data = results.multi_hand_landmarks[0].landmark  # Get first hand's landmarks
        index_tip = landmark_data[8]  # Index finger tip
        thumb_tip = landmark_data[4]  # Thumb tip

        # Get mouse position based on the index finger
        mouse_pos = get_mouse_pos_from_landmark_data(index_tip)

        # Move the mouse pointer
        if prev_mouse_pos is not None:
            mouse.move(mouse_pos[0] - prev_mouse_pos[0], mouse_pos[1] - prev_mouse_pos[1])

        # Check if clicking
        if is_clicking(thumb_tip, index_tip):
            if not is_dragging:
                mouse.press(Button.left)  # Press the left mouse button
                is_dragging = True
        else:
            if is_dragging:
                mouse.release(Button.left)  # Release the left mouse button
                is_dragging = False

        prev_mouse_pos = mouse_pos  # Update previous mouse position

        # Draw landmarks for visualization
        draw_hand_landmarks(frame, landmark_data)

    cv2.imshow('Hand Gesture Mouse', frame)
    key = cv2.waitKey(1)
    return key != ord('q')  # Exit if 'q' is pressed

# Function to get mouse position from hand landmark
def get_mouse_pos_from_landmark_data(index_tip):
    x = int(index_tip.x * 640)  # Screen width
    y = int(index_tip.y * 480)  # Screen height
    return x, y

# Function to check if the user is clicking
def is_clicking(thumb_tip, index_tip):
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return distance < 0.05  # Adjust threshold as needed

# Function to draw hand landmarks on the frame
def draw_hand_landmarks(image, landmarks):
    for landmark in landmarks:
        x = int(landmark.x * 640)
        y = int(landmark.y * 480)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw each landmark

# Main loop for video capture and gesture processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not process_frame(frame):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
