import cv2
import mediapipe as mp
from pymongo import MongoClient


# MongoDB 연결 설정
client = MongoClient('mongodb://localhost:27017/')  # MongoDB 연결 URI
db = client['test_db']  # 사용할 데이터베이스
reports_collection = db['analysis_reports2']  # 'ANALYSIS_REPORTS' 테이블에 해당하는 컬렉션

# MediaPipe의 손 인식 모듈 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 손 인식 설정 (Webcam 입력만 처리)
hands = mp_hands.Hands(
    static_image_mode=False,  # 실시간 영상 처리
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 손목과 엄지손톱을 이용한 손 상태 확인 (손이 화면에 있는지 확인)
def detect_hand_state(landmarks, frame_height):
    if landmarks:
        wrist = landmarks[0]  # 손목

        # 손목 y좌표가 화면 높이의 일정 비율 이상이면 손이 화면 밖으로 올라갔다고 판단
        if wrist.y < 0.2:  # 임계값 (20% 화면 위로 올라갔을 때)
            return "Hand Raised"
        elif wrist.y > 0.8:  # 손목이 화면 아래쪽으로 내려갔을 때
            return "Hand Low"
        else:
            return "Hand Visible"
    return "No Hand Detected"

# 실시간 웹캠 영상 분석 함수 (손 인식 포함)
def analyze_hand():
    video_capture = cv2.VideoCapture(0)  # 웹캠을 사용할 경우

    hand_movements = []
    frame_height = None

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # 화면 높이 설정 (처음 한 번만)
        if frame_height is None:
            frame_height = frame.shape[0]

        # MediaPipe로 손 인식 처리
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                hand_state = detect_hand_state(landmarks.landmark, frame_height)
                hand_movements.append(hand_state)

                # 손 상태 표시
                cv2.putText(frame, f"Hand State: {hand_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('Hand State Detection', cv2.flip(frame, 1))

        # 'Esc' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return hand_movements

# 실시간 웹캠 손 인식 시작
hand_movements = analyze_hand()


