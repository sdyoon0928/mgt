# childabuse/models.py

from django.db import models

# 선택지를 모델에도 명시할 수 있음 (추후 활용 가능)
GENDER_CHOICES = [('남아', '남아'), ('여아', '여아')]
ATTENDANCE_CHOICES = [('정상', '정상'), ('자주결석', '자주결석'), ('불규칙', '불규칙')]
NEG_LANG_CHOICES = [('낮음', '낮음'), ('중간', '중간'), ('높음', '높음')]
AGGRESSION_CHOICES = [('없음', '없음'), ('약함', '약함'), ('강함', '강함')]
CONTACT_CHOICES = [('선호', '선호'), ('중립', '중립'), ('회피', '회피'), ('공포', '공포')]
INCOME_CHOICES = [('낮음', '낮음'), ('중간', '중간'), ('높음', '높음')]
EMOTION_CHOICES = [('안정', '안정'), ('우울', '우울'), ('불안', '불안')]

class ChildObservation(models.Model):
    child_name = models.CharField(max_length=20)
    age = models.IntegerField()
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    attendance = models.CharField(max_length=20, choices=ATTENDANCE_CHOICES)
    negative_language = models.CharField(max_length=10, choices=NEG_LANG_CHOICES)
    parental_aggression = models.CharField(max_length=10, choices=AGGRESSION_CHOICES)
    contact_reaction = models.CharField(max_length=10, choices=CONTACT_CHOICES)
    sibling = models.IntegerField()  # 형제자매수
    income_level = models.CharField(max_length=10, choices=INCOME_CHOICES)
    emotional_state = models.CharField(max_length=10, choices=EMOTION_CHOICES)

    is_danger = models.BooleanField(default=False)
    reported = models.BooleanField(default=False)
    observation_date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.child_name} - {self.observation_date}"


class PredictionHistory(models.Model):
    child_name = models.CharField(max_length=20)
    predicted_result = models.CharField(max_length=100)
    predicted_prob = models.FloatField(default=0.0)  # ← 새 필드
    predicted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.child_name} - {self.predicted_result} ({self.predicted_at})"
