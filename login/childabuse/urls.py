from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_index, name='main_index'),  # ← 루트에 index.html을 연결
    path('dashboard/', views.home_view, name='home'),  # ← 대시보드 페이지를 /dashboard/로 이동
    path('predict/', views.predict_view, name='predict'),
    path('upload/', views.csv_upload_view, name='csv_upload'),
    path('form/', views.single_form_view, name='form_view'),
    path('bulk-form/', views.bulk_form_view, name='bulk_form_view'),
    path('reset/', views.reset_dashboard, name='reset_dashboard'),  # 추가
    
    
]