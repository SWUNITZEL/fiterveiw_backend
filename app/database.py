from pymongo import MongoClient
import json

# MongoDB 연결 설정
client = MongoClient('mongodb://localhost:27017/')
db = client['test_db']
reports_collection = db['analysis_reports']
contents_collection = db['report_contents']



# 문서 삽입
def insert_document(collection, document):
    collection.insert_one(document);

# 문서 검색
def find_document(collection, query):
    document = collection.find_one(query)
    return document


def update_video_analysis(video_json, q_num):
    try:

        latest_report = reports_collection.find().sort("rep_idx", -1).limit(1)

        if latest_report.count_documents({}) > 0:  # count_documents() 사용
            latest_rep_idx = latest_report[0]['rep_idx']

            # REPORT_CONTENTS 컬렉션에 분석 결과 업데이트
            update_result = contents_collection.update_one(
                {'rep_idx': latest_rep_idx, 'q_num': q_num},  # 조건: rep_idx와 q_num
                {'$set': {'video_analysis': video_json}}  # 업데이트할 내용
            )

            if update_result.modified_count > 0:
                print(f"rep_idx가 {latest_rep_idx}인 ANALYSIS_REPORTS 컬렉션에 답변 비디오 분석 결과 업데이트 성공")
            else:
                print(f"rep_idx가 {latest_rep_idx}인 REPORT_CONTENTS 컬렉션에 분석 결과 업데이트 실패")
        else:
            print("가장 최근의 rep_idx를 찾을 수 없습니다.")

    except Exception as e:
        print("Error while connecting to MongoDB", e)

    finally:
        print("DB 접속 종료")



# 예시 실행
video_json = json.dumps({"hand_state": "Hand Raised", "emotion": "Neutral"})  # 비디오 분석 결과 예시
q_num = 1  # 질문 번호 예시
update_video_analysis(video_json, q_num)


# DB 작업 후 연결 종료
def close_connection():
    client.close()
    print("MongoDB 접속 종료")

