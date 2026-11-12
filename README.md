# 신용점수가 놓친 28,211명: 거절 이력 대출자 재심사 모델링 (Home Credit)

과거 거절 이력만으로 자동 거절되던 대출 신청자가 정말 위험한지 데이터로 확인하고, 제안한 심사 정책을 과거 데이터에 백테스트해 검증한 팀 프로젝트입니다.

## 데이터
- Kaggle Home Credit Default Risk (공개 데이터)
  - https://www.kaggle.com/competitions/home-credit-default-risk
- 신청자 307,511명, 연결 테이블 7개(최대 약 2,700만 행), 전체 연체율 8.07%
- 원본 데이터는 용량 문제로 저장소에 포함하지 않았습니다. 위 링크에서 내려받아 `home-credit-default-risk/` 폴더에 두면 노트북을 실행할 수 있습니다.

## 분석 흐름
1. 7개 테이블을 신청자 단위로 먼저 집계한 뒤 병합해 196개 변수 생성 (신청자 1명당 1행이 되는지 행 수로 검증)
2. 통계 검정으로 연체자와 정상 고객의 차이 확인
3. K-Means 세분화로 거절 이력은 있지만 현재 신용점수가 양호한 회복형 고객 28,211명 식별
4. 조기경보 규칙과 가설 검정, 심슨의 역설 점검
5. LightGBM 예측 모델 (5-Fold 교차검증)
6. 정책 백테스트: 제안한 심사 정책을 과거 데이터에 소급 적용해 검증

## 핵심 결과
- 회복형 고객 28,211명의 실제 연체율은 6.83%로 전체 평균 8.07%보다 낮았습니다. 연령 등 교란 변수를 통제해도 결과가 유지됐습니다 (OR 0.868, p<1e-8).
- 모델 성능: 로지스틱 회귀 단일 테이블 기준선 AUC 0.7379 → LightGBM 7개 테이블 통합 AUC 0.7830
- 정책 백테스트
  - 처음 제안한 그룹별 승인 기준 완화는 같은 승인 규모에서 승인 집단의 연체자를 오히려 730명 늘렸습니다. 모델 점수가 이미 거절 이력을 반영하고 있어, 점수 위에 기준을 덧씌우면 고위험층이 들어오는 구조였습니다. 그래서 이 제안을 기각했습니다.
  - 대신 거절 이력에 따른 자동 거절만 없애고 점수로 심사하면, 회복형 고객의 승인율은 96.1%, 승인분 연체율은 5.99%로 평균 수준이었습니다.
  - 조기경보는 같은 경보 규모에서 규칙 기반(적중률 22.3%, 포착률 62.8%)보다 점수 기반(적중률 29.2%, 포착률 82.0%)이 우수했습니다.

## 한계
- 백테스트는 과거에 승인된 사람들의 데이터로만 가능합니다. 현재 거절 상태인 사람의 실제 상환 결과는 데이터에 없어(Reject Inference), 실제 적용 전에는 섀도우 테스트나 소규모 파일럿이 필요합니다.

## 파일 설명
| 파일 | 내용 |
|---|---|
| 03_feature_engineering_segmentation.ipynb | 7개 테이블 집계, 병합, 변수 생성, 세분화 |
| 04b_statistical_validation.ipynb | 통계 검정 |
| 05_act1_behavior_difference.ipynb | 연체자와 정상 고객의 행동 차이 |
| 06_act2_segmentation.ipynb | K-Means 세분화, 회복형 고객 식별 |
| 07_act3_early_warning.ipynb | 복합 리스크 점수, 조기경보 규칙 |
| 08_hypothesis_testing_full.ipynb | 가설 검정 종합 |
| 08_simpson_paradox_check.ipynb | 심슨의 역설 점검 |
| 09_ml_lightgbm.ipynb | LightGBM 예측 모델 |
| 10_ext_source_blindspot.ipynb | 외부 신용점수가 놓친 고객 분석 |
| 11_robustness_revision.ipynb | 과적합, DeLong 검정 등 강건성 보강 |
| 12_policy_backtest.ipynb | 정책 백테스트 |
| homecredit_erd.png | 테이블 관계도 |
| ml_performance_summary.csv, ml_feature_importance.csv | 모델 성능과 변수 중요도 |

## 사용 도구
Python, pandas, scikit-learn, LightGBM, scipy, statsmodels
