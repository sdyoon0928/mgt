// 로그인 모달 열기
document.getElementById('openLoginModal').onclick = function(event) {
  event.preventDefault();
  document.getElementById('modal').style.display = 'block';
};

// 로그인 모달 닫기
document.querySelector('#modal .close-modal').onclick = function() {
  document.getElementById('modal').style.display = 'none';
};

// 회원가입 모달 열기
document.querySelector('.signup-btn').onclick = function() {
  document.getElementById('modal').style.display = 'none'; // 로그인 모달 닫고
  document.getElementById('signupModal').style.display = 'block'; // 회원가입 모달 열기
};

// 회원가입 모달 닫기
document.querySelector('#signupModal .close-modal').onclick = function() {
  document.getElementById('signupModal').style.display = 'none';
};

// 바깥 클릭 시 모달 닫기 (로그인 + 회원가입)
window.onclick = function(event) {
  if (event.target == document.getElementById('modal')) {
    document.getElementById('modal').style.display = 'none';
  }
  if (event.target == document.getElementById('signupModal')) {
    document.getElementById('signupModal').style.display = 'none';
  }
};
