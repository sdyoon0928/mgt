# childabuse/forms.py

from django import forms
from .models import ChildObservation

class ObservationForm(forms.ModelForm):
    class Meta:
        model = ChildObservation
        fields = [
            'child_name', 'age', 'gender',
            'attendance', 'negative_language', 'parental_aggression',
            'contact_reaction', 'sibling', 'income_level', 'emotional_state'
        ]
    
    gender = forms.ChoiceField(
        choices=[('남아', '남아'), ('여아', '여아')],
        label='성별'
    )
    attendance = forms.ChoiceField(
        choices=[('정상', '정상'), ('자주결석', '자주결석'), ('불규칙', '불규칙')],
        label='출석 패턴'
    )
    negative_language = forms.ChoiceField(
        choices=[('낮음', '낮음'), ('중간', '중간'), ('높음', '높음')],
        label='부정 언어 표현'
    )
    parental_aggression = forms.ChoiceField(
        choices=[('없음', '없음'), ('약함', '약함'), ('강함', '강함')],
        label='보호자 공격성'
    )
        # 형제자매수 추가!
    sibling = forms.ChoiceField(
        label='형제자매 수',
        choices=[(i, str(i)) for i in range(6)]
    )

    # 신체접촉 반응 추가!
    contact_reaction = forms.ChoiceField(
        label='신체 접촉 반응',
        choices=[('중립', '중립'), ('회피', '회피'), ('공포', '공포')]
    )

    # 소득 수준 추가!
    income_level = forms.ChoiceField(
        label='소득 수준',
        choices=[('낮음', '낮음'), ('중간', '중간'), ('높음', '높음')]
    )

    # 보호자 정서 상태 추가!
    emotional_state = forms.ChoiceField(
        label='보호자 정서 상태',
        choices=[('안정', '안정'), ('불안', '불안'), ('우울', '우울')]
    )
    
    
        
        
class AbusePredictionForm(forms.Form):
    child_name = forms.CharField(label='아동 이름', max_length=30)
    age = forms.IntegerField(label='나이', min_value=0, max_value=18)
    gender = forms.ChoiceField(label='성별', choices=[('남아', '남아'), ('여아', '여아')])
    attendance = forms.ChoiceField(label='출석 패턴', choices=[('정상', '정상'), ('자주결석', '자주결석'), ('불규칙', '불규칙')])
    negative_language = forms.ChoiceField(label='부정 언어 표현', choices=[('낮음', '낮음'), ('중간', '중간'), ('높음', '높음')])
    parental_aggression = forms.ChoiceField(label='보호자 공격성', choices=[('없음', '없음'), ('약함', '약함'), ('강함', '강함')])
    contact_reaction = forms.ChoiceField(label='신체 접촉 반응', choices=[('중립', '중립'), ('회피', '회피'), ('공포', '공포')])
    sibling = forms.IntegerField(label='형제자매 수', min_value=0, max_value=5, initial=0)
    income_level = forms.ChoiceField(label='소득 수준', choices=[('낮음', '낮음'), ('중간', '중간'), ('높음', '높음')])
    emotional_state = forms.ChoiceField(label='보호자 정서 상태', choices=[('안정', '안정'), ('우울', '우울'), ('불안', '불안')])

