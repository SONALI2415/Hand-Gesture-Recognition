import cv2
import mediapipe as mp

model_path = "models/hand_landmarker.task"

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

def recognize_gesture(hand):
    index_tip = hand[8]
    index_pip = hand[6]

    middle_tip = hand[12]
    middle_pip = hand[10]

    ring_tip = hand[16]
    ring_pip = hand[14]

    pinky_tip = hand[20]
    pinky_pip = hand[18]

    index_open = index_tip.y < index_pip.y
    middle_open = middle_tip.y < middle_pip.y
    ring_open = ring_tip.y < ring_pip.y
    pinky_open = pinky_tip.y < pinky_pip.y

    finger_count = sum([
        index_open,
        middle_open,
        ring_open,
        pinky_open
    ])

    if finger_count == 0:
        return "FIST"

    elif finger_count == 4:
        return "OPEN PALM"

    elif finger_count == 1 and index_open:
        return "ONE"

    elif finger_count == 2 and index_open and middle_open:
        return "TWO"

    elif finger_count == 3 and index_open and middle_open and ring_open:
        return "THREE"

    else:
        return "UNKNOWN"


def get_action(gesture):
    if gesture == "FIST":
        return "PAUSE"

    elif gesture == "OPEN PALM":
        return "PLAY"

    elif gesture == "ONE":
        return "VOLUME UP"

    elif gesture == "TWO":
        return "VOLUME DOWN"

    elif gesture == "THREE":
        return "NEXT"

    else:
        return "NO ACTION"


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not open camera")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    frame_number = 0

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = frame_number * 33

        result = landmarker.detect_for_video(
            image,
            timestamp
        )

        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                height, width, _ = frame.shape

                gesture = recognize_gesture(hand)

                action = get_action(gesture)

                for point in hand:

                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                cv2.putText(
                    frame,
                    "Gesture: " + gesture,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Action: " + action,
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2
                )

        else:

            cv2.putText(
                frame,
                "No hand detected",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Action: NO ACTION",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "Q = Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Hand Gesture Recognition",
            frame
        )

        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()