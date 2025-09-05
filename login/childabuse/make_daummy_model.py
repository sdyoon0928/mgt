# childabuse/make_dummy_model.py

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 임의의 훈련 데이터
data = pd.DataFrame({
    '나이': [5, 6, 7],
    '성별': [0, 1, 0],
    '출석패턴': [0, 1, 2],
    '부정언어표현': [1, 0, 2],
    '보호자공격성': [0, 2, 1]
})
target = [0, 1, 0]

# 모델 학습 및 저장
model = RandomForestClassifier()
model.fit(data, target)

# 저장 경로 생성
import os
os.makedirs('childabuse/model', exist_ok=True)
joblib.dump(model, 'childabuse/model/random_forest_model.pkl')
