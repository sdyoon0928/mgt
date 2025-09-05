# childabuse/admin.py

from django.contrib import admin
from .models import ChildObservation, PredictionHistory

@admin.register(ChildObservation)
class ChildObservationAdmin(admin.ModelAdmin):
    list_display = [
        'child_name', 'age', 'gender', 'attendance',
        'negative_language', 'parental_aggression',
        'is_danger', 'reported', 'observation_date'
    ]
    list_filter = ['gender', 'attendance', 'parental_aggression', 'is_danger', 'reported']

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['child_name', 'predicted_result', 'predicted_at']
