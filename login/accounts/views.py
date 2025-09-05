from django.shortcuts import redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages

# 회원가입 뷰
def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('name')         # 사용자가 입력한 이름
        email = request.POST.get('email')
        password = request.POST.get('password')

        if User.objects.filter(email=email).exists():
            messages.error(request, '이미 등록된 이메일입니다.')
            return redirect('/')

        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)  # 회원가입 후 자동 로그인

        # 디버그 출력
        print("=== 회원가입 디버깅 ===")
        print("user.is_active:", user.is_active)
        print("user.is_authenticated:", user.is_authenticated)
        print("user.username:", user.username)

        messages.success(request, f'{username}님, 가입을 축하합니다!')
        return redirect('/')

    return redirect('/')

# 로그인 뷰
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = User.objects.get(email=email)
            user = authenticate(request, username=user.username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'{user.username}님, 환영합니다!')
            else:
                messages.error(request, '비밀번호가 올바르지 않습니다.')
        except User.DoesNotExist:
            messages.error(request, '해당 이메일로 등록된 사용자가 없습니다.')

    return redirect('/')

# 로그아웃 뷰
def logout_view(request):
    logout(request)
    messages.success(request, '로그아웃이 성공적으로 완료되었습니다.')
    return redirect('/')
