# childabuse/views.py

import os
import json
import joblib
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from .models import ChildObservation, PredictionHistory
from .forms import ObservationForm, AbusePredictionForm


# 0. 누적 테이블 초기화
@require_POST
def reset_dashboard(request):
    ChildObservation.objects.all().delete()
    PredictionHistory.objects.all().delete()
    return redirect('home')



# ✅ 1. CSV 파일 경로 설정
CSV_PATH = os.path.join(settings.BASE_DIR, 'childabuse', 'data', '현실기반_아동학대_더미데이터_500건.csv')
DATA_PATH = os.path.join(settings.BASE_DIR, 'childabuse', 'data', '어린이집_더미데이터_확장본_250326.csv')
MODEL_PATH = os.path.join(settings.BASE_DIR, 'childabuse', 'model', 'random_forest_model.pkl')
META_PATH = os.path.join(settings.BASE_DIR, 'childabuse', 'model', 'model_meta.json')


# ✅ 모델 로드
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle['model']
accuracy = round(model_bundle['accuracy'] * 100, 2)
model_accuracy = f"정확도: {round(model_bundle['accuracy'] * 100, 2)}%"

df = pd.read_csv(CSV_PATH)

# 문자형 → 수치형 매핑
df = df.replace({
    '성별': {'남': 0, '남아': 0, '여': 1, '여아': 1},
    '출석패턴': {'정상': 0, '자주결석': 1, '불규칙': 2},
    '부정언어표현': {'낮음': 0, '중간': 1, '높음': 2},
    '보호자공격성': {'없음': 0, '약함': 1, '강함': 2},
    '신체접촉반응': {'선호': 0, '중립': 1, '회피': 2, '공포': 3},
    '소득수준': {'낮음': 0, '중간': 1, '높음': 2},
    '보호자정서상태': {'안정': 0, '우울': 1, '불안': 2}
})

# 과거신고이력은 별도로 강제 처리
df['과거신고이력'] = df['과거신고이력'].replace({'없음': 0, '있음': 1}).astype(int)

# 오류 방지용 형변환
df = df.infer_objects(copy=False)

# 평균 계산
mean_values = df[[
    '나이', '성별', '출석패턴', '부정언어표현',
    '보호자공격성', '신체접촉반응', '형제자매수',
    '소득수준', '보호자정서상태'
]].mean().round(2).tolist()


# ✅ 매핑 딕셔너리
GENDER_MAP = {'남아': 0, '여아': 1}
ATTENDANCE_MAP = {'정상': 0, '자주결석': 1, '불규칙': 2}
NEG_LANG_MAP = {'낮음': 0, '중간': 1, '높음': 2}
AGGRESSION_MAP = {'없음': 0, '약함': 1, '강함': 2}
REACTION_MAP = {'선호': 0, '중립': 1, '회피': 2, '공포': 3}
INCOME_MAP = {'낮음': 0, '중간': 1, '높음': 2}
EMOTION_MAP = {'안정': 0, '우울': 1, '불안': 2}


# ✅ 예측 함수
def predict_danger(instance):
    input_df = pd.DataFrame([{
        '나이': instance.age,
        '성별': GENDER_MAP.get(instance.gender, 0),
        '출석패턴': ATTENDANCE_MAP.get(instance.attendance, 0),
        '부정언어표현': NEG_LANG_MAP.get(instance.negative_language, 1),
        '보호자공격성': AGGRESSION_MAP.get(instance.parental_aggression, 1),
        '신체접촉반응': REACTION_MAP.get(instance.contact_reaction, 1),
        '형제자매수': instance.sibling,
        '소득수준': INCOME_MAP.get(instance.income_level, 1),
        '보호자정서상태': EMOTION_MAP.get(instance.emotional_state, 0)
    }])
    prob = model.predict_proba(input_df)[0][1]
    return (prob >= 0.5), round(prob * 100, 2)


def predict_danger_extended(data):
    input_df = pd.DataFrame([{
        '나이': int(data.get('age', 0)),
        '성별': GENDER_MAP.get(data.get('gender'), 0),
        '출석패턴': ATTENDANCE_MAP.get(data.get('attendance'), 0),
        '부정언어표현': NEG_LANG_MAP.get(data.get('negative_language'), 1),
        '보호자공격성': AGGRESSION_MAP.get(data.get('parental_aggression'), 1),
        '신체접촉반응': REACTION_MAP.get(data.get('contact_reaction'), 1),
        '형제자매수': int(data.get('sibling', 0)),
        '소득수준': INCOME_MAP.get(data.get('income_level'), 1),
        '보호자정서상태': EMOTION_MAP.get(data.get('emotional_state'), 0)
    }])
    prediction = model.predict(input_df)[0]
    return prediction, input_df.iloc[0].tolist()


# ✅ 홈
def home_view(request):
    form = ObservationForm(request.POST or None)
    
    if request.method == 'POST':
        print("📩 폼 제출 감지됨")
        print("✅ 유효성 검사 통과 여부:", form.is_valid())
        print("🚨 에러:", form.errors)
    
        if request.method == 'POST' and form.is_valid():
            instance = form.save(commit=False)  # 저장은 잠시 보류하고
            is_danger, prob = predict_danger(instance)  # 예측 수행
            instance.is_danger = is_danger
            instance.save()  # 예측 결과 포함 저장

            PredictionHistory.objects.create(
                child_name=instance.child_name,
                predicted_result="위험" if is_danger else "정상",
                predicted_prob=prob  # 이 줄 꼭 있어야 그래프에도 뜹니다!
            )
            return redirect('home')  # 홈으로 돌아와 리스트에 반영
        
    # 관찰 리스트 및 예측 결과 리스트 불러오기
    observations = ChildObservation.objects.all().order_by('-observation_date')
    predictionhistory_list = PredictionHistory.objects.all()
    context = {
        'form': form,
        'observation_list': observations,
        'predictionhistory_list': predictionhistory_list,
        'total_kids': len(df),
        'danger_kids': int(df['과거신고이력'].sum()),
        'last_report_date': f"정확도: {accuracy}%"
    }
    return render(request, 'main_home.html', context)

# ✅ 단건 예측 (/predict)
def predict_view(request):
    form = AbusePredictionForm(request.POST or None)
    context = {'form': form, 'range_0_5': range(6)}

    if request.method == 'POST' and form.is_valid():
        data = request.POST.dict()
        prediction, input_values = predict_danger_extended(data)
        is_danger = prediction == 1

        instance = ChildObservation(
            child_name=data.get('child_name'),
            age=int(data.get('age')),
            gender=data.get('gender'),
            attendance=data.get('attendance'),
            negative_language=data.get('negative_language'),
            parental_aggression=data.get('parental_aggression'),
            contact_reaction=data.get('contact_reaction'),
            sibling=int(data.get('sibling', 0)),
            income_level=data.get('income_level'),
            emotional_state=data.get('emotional_state'),
            is_danger=is_danger
        )
        instance.save()

        PredictionHistory.objects.create(
            child_name=instance.child_name,
            predicted_result="위험" if is_danger else "정상"
        )

        context.update({
            'result': f"예측 결과: {'위험' if is_danger else '정상'}",
            'accuracy': f"정확도: {accuracy}%",
            'input_values': input_values,
            'feature_means': mean_values
        })
    elif request.method == 'POST':
        context['result'] = "❌ 유효하지 않은 입력입니다."

    return render(request, 'form.html', context)


# ✅ 입력 폼 및 업로드 뷰들
def main_index(request):
    return render(request, 'index.html')

def single_form_view(request):
    form = AbusePredictionForm()  # 🔥 폼 객체 생성
    context = {'form': form, 'range_0_5': range(6)}  # 🔥 context에 form 전달
    return render(request, 'form.html', context)

def bulk_form_view(request):
    return render(request, 'bulk_form.html')


def csv_upload_view(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            ChildObservation.objects.create(
                child_name=row['아동이름'],
                age=row['나이'],
                gender=row['성별'],
                attendance=row['출석'],
                negative_language=row['부정언어표현'],
                parental_aggression=row['보호자공격성'],
                contact_reaction=row.get('신체접촉반응', '중립'),
                sibling=row.get('형제자매수', 0),
                income_level=row.get('소득수준', '중간'),
                emotional_state=row.get('보호자정서상태', '안정'),
                is_danger=row.get('is_danger', False),
            )
        return redirect('home')
    return render(request, 'bulk_form.html')
