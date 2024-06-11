# 설치

- Anaconda를 통한 Full-dependency 보장을 선호함
- python 3.12 버전 설치 완료
  - '/opt/homebrew/opt/python@3.12/bin/python3.12'
- sh 파일은 $sh filename.sh 로 실행 가능
- Anaconda 설치 완료
- pip 설치 완료

> 알고보니 Mac에서 자동으로 제공되는 Python 3.9 버전이 Homebrew 밖에 설치되어 있었고,  
> 없는 줄 알고 Homebrew로 3.12버전 설치  
> 3.12 버전에 Pytorch 설치 완료  
> 고민하다가, Python 가상환경 .conda(3.9ver) 을 이용해 보기로 결정함.  

- 가상환경을 이용하면, 터미널 영역에서 패키지를 설치했을 때, .conda 혹은 .venv 폴더의 site-packages 폴더 안에 설치가 된다.
- 가상환경을 이용하면, 파이썬 버전관리에 용이하다.
  - 아버지 알리미에서 pyinstaller 버전 때문에 고생했던 걸 생각하면, 유지 보수 측면에서 가상환경을 만들어 작업하는게 우월전략인 듯 하다.
  - 이렇게 개발해서 만든 모델만 실어서 서비스를 런칭하는 방식을 생각해 볼 수 있겠다.
- 가상환경을 삭제하고 싶다면, 그냥 .venv, .conda 폴더를 삭제해주면 된다.

# 의문점

- venv 가상환경과 conda 가상환경의 차이점?
  - venv 에서는 당연하게도 pip 명령어를 사용해 패키지를 설치하고,
  - conda 에서는 conda 명령어를 사용해 설치한다.
  - venv는 python 3.x 이후 내장, conda는 python이 아닌 다른 언어도 지원한다.
  - conda 에는 자동으로 패키지들의 버전관리를 해주는 기능이 있다. (전체 업데이트 등)
  - 머신러닝, 복잡한 의존성의 프로젝트에는 conda가.
  - 간단한 python 프로젝트에는 venv가 적절하다.

