# main.py (학습용 스크립트로 변경)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 데이터 불러오기
df = pd.read_csv("현실기반_아동학대_더미데이터_500건.csv")

# 2. 매핑 처리
mapping_dict = {
    '성별': {'남': 0, '여': 1, '남아': 0, '여아': 1},
    '출석패턴': {'정상': 0, '자주결석': 1, '불규칙': 2},
    '부정언어표현': {'낮음': 0, '중간': 1, '높음': 2},
    '보호자공격성': {'없음': 0, '약함': 1, '강함': 2},
}
for col, mapper in mapping_dict.items():
    df[col] = df[col].astype(str).str.strip().replace(mapper)

# 3. 특성과 라벨 설정
features = ['나이', '성별', '출석패턴', '부정언어표현', '보호자공격성']
X = df[features]
y = (df['보호자공격성'] >= 2).astype(int)  # 학대 위험 여부

# 4. 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 5. 모델 저장 (.pkl 파일)
joblib.dump(model, 'childabuse/model/random_forest_model.pkl')
print("🎉 모델 저장 완료: childabuse/model/random_forest_model.pkl")
