import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,  # 최대 1개의 얼굴을 인식
                                 refine_landmarks=True,  # 동공 위치 확인
                                 min_detection_confidence=0.6,  # 최소 감지 확신도
                                 min_tracking_confidence=0.7)  # 최소 추적 확신도
mp_drawing = mp.solutions.drawing_utils

# 주요 랜드마크 인덱스
LEFT_IRIS = [474, 475, 476, 477]  # 왼쪽 홍채
RIGHT_IRIS = [469, 470, 471, 472]  # 오른쪽 홍채
LEFT_EYE = [33, 159, 145, 133]  # 왼쪽 눈 (위쪽, 아래쪽 포함)
RIGHT_EYE = [362, 386, 374, 263]  # 오른쪽 눈 (위쪽, 아래쪽 포함)

# 웹캠 활성화
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 효과)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape  # 프레임 크기

            # 홍채 중심 좌표 계산
            left_iris_x = np.mean([face_landmarks.landmark[i].x for i in LEFT_IRIS]) * w
            left_iris_y = np.mean([face_landmarks.landmark[i].y for i in LEFT_IRIS]) * h
            right_iris_x = np.mean([face_landmarks.landmark[i].x for i in RIGHT_IRIS]) * w
            right_iris_y = np.mean([face_landmarks.landmark[i].y for i in RIGHT_IRIS]) * h

            # 눈의 중심 좌표 계산
            left_eye_x = np.mean([face_landmarks.landmark[i].x for i in LEFT_EYE]) * w
            left_eye_y = np.mean([face_landmarks.landmark[i].y for i in LEFT_EYE]) * h
            right_eye_x = np.mean([face_landmarks.landmark[i].x for i in RIGHT_EYE]) * w
            right_eye_y = np.mean([face_landmarks.landmark[i].y for i in RIGHT_EYE]) * h

            # 기준값 설정
            threshold_y = 3.4  # 상하 이동 감지 임계값

            gaze_direction = ""

            # 왼쪽 눈 기준으로 시선 판별
            if left_iris_y < left_eye_y - threshold_y:
                gaze_direction = "Looking Up"
            elif left_iris_y > left_eye_y + threshold_y:
                gaze_direction = "Looking Down"
            else:
                gaze_direction = "Looking Center"

            # # 양쪽 눈 기준으로 시선 판별
            # if (left_iris_y + right_iris_y) / 2 < (left_eye_y + right_eye_y) / 2 - threshold_y:
            #     gaze_direction = "Looking Up"
            # elif (left_iris_y + right_iris_y) / 2 > (left_eye_y + right_eye_y) / 2 + threshold_y:
            #     gaze_direction = "Looking Down"
            # else:
            #     gaze_direction = "Looking Center"

            # 화면에 출력
            cv2.putText(frame, gaze_direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 홍채 및 눈 랜드마크 시각화
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES)

    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
