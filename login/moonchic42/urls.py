from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # 메인 홈화면 → childabuse 내부에서 메인 페이지 처리
    path('', include('childabuse.urls')),  

    # 각 앱 URL 연결
    path('accounts/', include('accounts.urls')),
    path('inventory/', include('inventory.urls')),
]

# 정적 파일 서빙 (개발 모드)
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
