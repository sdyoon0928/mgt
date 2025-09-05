import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# 데이터 로드
DATA_PATH = 'childabuse/data/어린이집_더미데이터_확장본_250326.csv'
df = pd.read_csv(DATA_PATH)

# 문자 → 숫자 매핑
df = df.replace({
    '성별': {'남': 0, '여': 1, '남아': 0, '여아': 1},
    '출석패턴': {'정상': 0, '자주결석': 1, '불규칙': 2},
    '부정언어표현': {'낮음': 0, '중간': 1, '높음': 2},
    '보호자공격성': {'없음': 0, '약함': 1, '강함': 2},
    '신체접촉반응': {'선호': 0, '중립': 1, '회피': 2, '공포': 3},
    '소득수준': {'낮음': 0, '중간': 1, '높음': 2},
    '보호자정서상태': {'안정': 0, '우울': 1, '불안': 2},
    '과거신고이력': {'없음': 0, '있음': 1}
})

features = ['나이', '성별', '출석패턴', '부정언어표현', '보호자공격성',
            '신체접촉반응', '형제자매수', '소득수준', '보호자정서상태']
X = df[features]
y = df['과거신고이력']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"✅ 모델 정확도: {accuracy:.2%}")

os.makedirs('childabuse/model/', exist_ok=True)
joblib.dump({'model': model, 'accuracy': accuracy},
            'childabuse/model/random_forest_model.pkl')  # ✅ 정확도 포함 저장