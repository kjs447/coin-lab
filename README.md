# Coin Lab

FastAPI + Backtrader 연구 환경으로, 업비트 60분 봉 데이터를 수집하고 여러 전략을 실험할 수 있습니다. 프로젝트는 로컬 `venv` 가상환경을 기반으로 실행하도록 구성되어 있습니다.

## 요구 사항

- Python 3.10 이상 (시스템에 `python3` 명령이 있어야 합니다)
- 업비트 OpenAPI 키 없이도 캔들 조회는 가능하지만, 호출 한도 정책을 확인하세요.

## 가상환경 설정

```bash
scripts/setup_env.sh
```

스크립트는 `.venv` 디렉터리에 가상환경을 만들고 `requirements.txt` 를 설치합니다. 이후 가상환경을 사용하려면:

```bash
source .venv/bin/activate
```

## 서버 실행

가상환경이 준비된 상태에서 다음을 실행합니다.

```bash
scripts/run_server.sh
```

기본적으로 `http://127.0.0.1:8000` 에서 FastAPI 앱이 실행되며, `/` 경로에서 실험 페이지를 확인할 수 있습니다. 코드 변경 시 자동 재시작(`--reload`)이 활성화되어 있습니다.

## 기타 명령

가상환경을 수동으로 활성화한 뒤 사용할 수 있는 명령 예시는 다음과 같습니다.

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir .
python -m compileall app  # 정적 검사
```

## 데이터베이스

기본 데이터베이스는 프로젝트 루트의 `coin_lab.db` SQLite 파일입니다. 환경 변수 `COIN_LAB_DATABASE_URL` 을 설정하면 다른 DB로 전환 가능합니다.
