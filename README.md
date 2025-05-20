본 레포지토리는 Neural Collaborative Filtering(NCF) 모델을 TensorFlow 2.0 및 Keras 기반으로 구현한 것입니다.  
MovieLens 1M 데이터셋을 사용하여 사용자-아이템 평점을 예측하는 회귀 문제로 정의하였으며, 원 논문과 Keras 공식 예제 및 블로그 구현을 참고하였습니다.

---

## 파일 설명

- `main.py` : 전체 실행 및 학습/평가 흐름 제어
- `data/preprocess.py` : MovieLens 데이터 전처리 및 train/test 분리
- `models/ncf_keras.py` : Keras 기반 NCF 모델 정의
- `utils/metrics.py` : RMSE, MAE 평가 지표 계산 함수
- `requirements.txt` : pip 기반 의존성 패키지 목록

---

## 데이터

- 사용 데이터: `ratings.dat` (MovieLens 1M)
- 다운로드 경로: https://grouplens.org/datasets/movielens/1m/
- 파일 위치: `data/ratings.dat`에 저장

---

## 코드 실행 예시
```python
python main.py
```

## Reference
1. He, Xiangnan, et al. "Neural Collaborative Filtering." Proceedings of the 26th International Conference on World Wide Web. 2017.
2. https://github.com/hexiangnan/neural_collaborative_filtering

## Update
Last Update Date: 2025/05/20