import cv2
from cvzone.HandTrackingModule import HandDetector

# OpenCV setup
cap = cv2.VideoCapture(0)
cv2.namedWindow("Finger Counter", cv2.WINDOW_NORMAL)

# Hand detector setup
detector = HandDetector(detectionCon=0.8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Find hands in the frame
    hands, frame = detector.findHands(frame)

    if hands:
        hand = hands[0]  # Assuming only one hand in the frame

        # Get the landmarks of the hand
        landmarks = hand["lmList"]

        # Count fingers and display the count
        finger_count = detector.fingersUp(hand)
        cv2.putText(frame, f"Fingers: {sum(finger_count)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Finger Counter", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
