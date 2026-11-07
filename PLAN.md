# XAIParkinson 실행 계획

최종 수정: 2026-11-07 (계획 단계. 다운로드·설치·학습은 하지 않음)
원자료와 링크: `NOTES.md`

## 0. 목표와 기본 경로
- **주장**: CNN 기반 파킨슨 탐지 SOTA 모델은 정확도는 높지만 설명가능하지 않다. 근거로 다음을 실험에서 보인다.
  - Grad-CAM이 임상적으로 의미 없는 영역, 배경, 아티팩트에 주목한다.
  - 모델이나 시드가 바뀌면 설명이 달라진다.
  - sanity check를 통과하지 못한다.
  - shortcut learning의 흔적이 있다.
- **기본 경로 (사용자 지시)**: 직접 학습하지 않는다. 다음 순서로 진행한다.
  1. 공개된 사전학습 체크포인트를 받는다.
  2. 그 모델의 학습 데이터와 같은 공개 테스트셋에서 정확도를 먼저 확인한다.
  3. 그다음 Grad-CAM 계열 분석으로 넘어간다.
- 선정 기준
  1. 체크포인트가 공개돼 있는가
  2. 같은 공개 데이터셋(테스트셋)이 있는가
  3. CNN인가
  4. 안전한 형식인가 (safetensors > h5/onnx > pth(weights_only) > pickle)
  5. 라이선스

---

## 1. 모달리티별 SOTA 요약 (자세한 표는 NOTES.md 3절)
| 모달리티 | 대표 SOTA (보고 성능) | 공개 가중치 | 기존 XAI |
|---|---|---|---|
| 나선·물결 그리기 (Kaggle) | VGG16/19 TL 약 90% (arXiv 2510.05015), CNN 앙상블 약 93% (Chakraborty 2020, 추정) | **있음** (아래 2절, 다만 비공식 구현) | 정성적 Grad-CAM만 있다 (ParkXplainer 2025) |
| HandPD/NewHandPD | CNN 94.7–100% (Bensefia 2026), VGG16 95% (IJPRAI 2024) | 없음 | Grad-CAM (정성) |
| PaHaW | 동역학 특징 + SVM (CNN 아님) | 없음 | — |
| DaTscan (PPMI 등) | VGG16 95.2% (Magesh 2020), 2D CNN 98.5% (2024), DaT Challenge 2026 log loss 0.25 | **있음** (DaT Challenge 1위, MIT/Apache) | LIME (Magesh), Grad-CAM/SHAP (Pianpanit 2021) |
| MRI (T1, NM-MRI) | NM-MRI ResNet50 80% (Shinde 2019), T1 3D CNN 84.5% (2025) | 카드 없는 HF Keras 모델뿐 → 제외 | saliency |
| 음성 | Italian PVS CNN 87–99% (검증 방식이 제각각) | 찾지 못함 | 거의 없음 |

## 2. 공개 가중치 후보 (선정 결과)
| 순위 | 모델 | 가중치 | 형식 / 위험 | 같은 공개 테스트셋 | 비고 |
|---|---|---|---|---|---|
| A1 | Parkonix (Teachable Machine, MobileNet 계열 추정) | GitHub `keras_model.h5` | Keras h5 / 중간 (Lambda 검사 필요) | Kaggle spiral/wave test | 83%라고 주장. 학습 split은 미상 |
| A2 | sayakpaul ResNet-34 (fastai) | GitHub `models/parkinson-predictor.pkl` 43.8MB | **pickle / 높음** | **저장소 안에 학습에 쓴 train/test 폴더가 그대로 있다** → 정확히 같은 테스트셋으로 확인할 수 있다 | fastai v1이라 격리된 구버전 환경이 필요하다 |
| A3 | gianlab Swin-Tiny (비교군, CNN 아님) | HF `model.safetensors` 110MB | safetensors / 낮음 | Kaggle spiral (eval split 미공개) | 형식은 가장 안전하다. "CNN vs ViT" 대조군으로 쓴다 |
| B1 | DaT Challenge 1위 Siamese CNN 앙상블 | HF `dinhviettoanle/dat-parkinson-models` .pth 100개, 약 788MB | pth / weights_only로 로드 | DaT 대회 데이터 (DrivenData 로그인, 대회 후 접근 미확인) 또는 PPMI (외부 검증) | 추론 스크립트 있음. 입력이 선조체 crop |
| B2/B3 | DaT Challenge 2위 블렌드 / 3위 DenseNet121 | 저장소 스크립트 (위치 확인 필요). 2위는 SHA256SUMS 제공 | pth 추정 | 위와 같음 | 3위는 전체 뇌 2D 투영 → 선조체 ROI 실험에 가장 알맞다 (확인 필요) |
| 제외 | dhhd255 EfficientNet (헤드 없음), DhruvW .pth·yuchenlung .h5 (카드 없음), HF MRI Keras 모델들 (카드 없음), neelanjan00 (sklearn pickle) | — | — | — | 문서가 없어 재현성을 확인할 수 없다 |

## 3. 1차 타깃 추천
### 타깃 1 (지금 바로 시작): 나선·물결 그리기 = Kaggle kmader / PyImageSearch 데이터 + A1·A2·A3
- **근거**
  - 공개 가중치가 3개 이상 있다.
  - 같은 이미지가 sayakpaul 저장소에 들어 있어 로그인 없이도 받을 수 있다. Kaggle 원본도 있다.
  - 데이터가 약 20MB(추정)이고 추론은 CPU로도 몇 초면 끝난다.
  - 같은 테스트셋 60장(spiral 30 + wave 30)으로 여러 아키텍처를 비교할 수 있다 → 아키텍처 간 히트맵 일관성 실험에 쓴다.
  - 배경이 흰 종이와 스캔 아티팩트라서 "획 밖에 주목하는 비율"을 정량화하기 쉽다(획 마스크는 임계값으로 자동 생성).
  - 선행 근거가 있다. FernandoPCJ 감사에서 1-NN만으로 AUC 0.94–0.96이 나왔고, 물결 Grad-CAM은 획에 집중하지 않았다.
- **추가 실험 (학습 불필요)**: 같은 모델을 NewHandPD·HandPD 나선(UNESP, 바로 다운로드)에 zero-shot으로 적용한다. 정확도가 크게 떨어지면 원 데이터셋에서 shortcut을 배웠다는 근거가 된다.
- **사용자 할 일**: 없음. Kaggle 원본을 쓰려면 Kaggle 계정과 `kaggle.json` API 토큰이 필요하다(선택).

### 타깃 2 (승인 대기 중 병행): DaTscan = DaT Challenge 1위(+2·3위) 가중치 + 테스트 데이터
- **근거**
  - 임상 ROI(선조체: 미상핵·피각)가 명확해 IoU와 pointing game을 의미 있게 계산할 수 있다.
  - 2026년 대회 상위 모델이라 "SOTA"라고 부를 근거가 강하다.
  - PyTorch, MIT/Apache 라이선스, 추론 스크립트가 있다.
- **주의**
  - 1위 모델은 선조체 crop을 입력으로 받는다. 그래서 "선조체 밖 주목"은 crop 안 배경과 좌우 비대칭 축에서만 볼 수 있다.
  - 전체 뇌를 입력으로 받는 3위(투영 DenseNet121)를 같이 확인한다.
- **사용자 할 일 (둘 중 하나 이상)**
  1. **DrivenData 계정**: 대회 페이지(https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)에 로그인해 대회 종료 후에도 data 탭에서 받을 수 있는지와 약관을 확인한다. 약관에 동의하는 것은 사용자가 직접 한다.
  2. **PPMI DUA**: https://www.ppmi-info.org/access-data-specimens/download-data 에서 신청한다. 개인정보, 소속 기관, 분석 목적을 적고 DUA에 동의한다. 검토는 약 1주가 목표다. 승인되면 LONI IDA에서 DaTscan "Reconstructed/Processed" 시리즈를 받는다. 이 경우 대회 데이터와 PPMI가 겹치는지 확인해야 한다(겹치면 누수).
  - 미성년자나 고등학생이 신청하면 기관 소속 요건 때문에 지도교사나 연구기관 명의가 필요할 수 있다 [추정].

### 타깃 3 (최후 대안, 최소한만): NewHandPD 나선 + ImageNet ResNet18 짧은 fine-tuning
- 타깃 1의 가중치를 모두 로드하지 못하거나 누수 때문에 쓸 수 없을 때만 한다.
- RTX 4070에서 5–15분 정도(추정). 시드 3개 × 아키텍처 2개까지만 한다.
- 이 경로도 "공개 SOTA 레시피 재현"이지 "SOTA 가중치"가 아니라는 점을 보고서에 밝힌다.

음성과 MRI는 공개 가중치와 공개 데이터 조합이 없어 이번 범위에서 제외한다(NOTES 1.3, 1.4절).

---

## 4. 실험 설계 (학습 없이)
| # | 실험 | 방법 | 뒷받침하는 주장 |
|---|---|---|---|
| E0 | 정확도 확인 | 공개 가중치로 같은 테스트셋에서 acc/AUC/sens/spec를 계산하고 보고값과 비교한다. 전처리(크기, 정규화, 채널, 라벨 순서)를 모델 카드나 코드와 맞춘다 | "정확도는 높다"는 전제를 우리 손으로 확인한다. 보고값보다 낮으면 그 자체가 재현성 문제의 근거다 |
| E1 | Grad-CAM / Grad-CAM++ / Score-CAM 시각화 | pytorch-grad-cam으로 마지막 conv 블록에 적용한다. 정답·오답, 클래스별로 나눠 본다 | 정성 근거 (그림) |
| E2 | 배경·아티팩트 주목 비율 | 그리기: 획 마스크(이진화 + 팽창) 밖 CAM 에너지 비율, 그리고 획 마스크 크기를 고려한 enrichment. DaTscan: 뇌 밖이나 선조체 밖 비율 | 모델이 임상적으로 무의미한 영역을 본다 |
| E3 | 임상 ROI 겹침 | DaTscan: 선조체 아틀라스(미상핵·피각) 대비 상위 k% CAM의 IoU와 pointing game(최대값이 ROI 안에 있는지). 그리기: 획 위 pointing game | 설명이 임상 지식과 맞지 않는다 |
| E4 | 아키텍처·시드 간 일관성 | A1·A2·A3(그리고 B1·B3)의 같은 입력 히트맵을 Spearman 순위상관, top-k IoU로 비교한다. 앙상블 멤버(DaT 1위의 fold별 모델)끼리도 비교한다(사실상 시드 차이) | 같은 정답을 내도 근거가 제각각이라 설명을 믿기 어렵다 |
| E5 | Adebayo sanity check | (a) 모델 무작위화: 위층부터 차례로 가중치를 재초기화하고 히트맵 변화를 SSIM·순위상관으로 측정한다. 학습이 필요 없다. (b) 라벨 무작위화: 재학습이 필요해서 **선택 사항**이다. 타깃 3의 짧은 학습 경로에서만 한다 | 히트맵이 모델이 배운 것과 무관하다(엣지 검출기 수준이다) |
| E6 | Deletion / Insertion | CAM 순위대로 픽셀을 지우거나 넣으면서 확률 곡선 AUC를 구한다. 무작위 순서, 획 마스크 순서 baseline과 비교한다 | 히트맵이 실제 결정에 충실하지 않다(faithfulness 부족) |
| E7 | Shortcut / 마스킹 테스트 | (a) 획만 남기고 배경을 흰색으로 지운 입력, (b) 반대로 획을 지운 배경만 남긴 입력, (c) 테두리·스캔 아티팩트만 남긴 입력의 정확도를 본다. FernandoPCJ식 1-NN baseline, near-duplicate 검사(pHash)도 한다 | 정확도가 질병 신호가 아니라 지름길에서 나온다 |
| E8 | 교차 데이터셋 (zero-shot) | 타깃 1 모델을 NewHandPD/HandPD 나선에 적용한다. DaT 모델은 PPMI에 적용한다 | 높은 정확도가 일반화되지 않는다 → shortcut 근거를 보탠다 |

통계: 이미지 단위 bootstrap 95% CI를 쓰고, 모델 간 비교에는 Wilcoxon 검정을 쓴다. 테스트셋이 60장으로 작으므로 모든 수치에 CI를 붙인다.

### 사전학습 가중치 사용에 따른 위험과 대응
| 위험 | 대응 |
|---|---|
| 학습 데이터 누수 (공개 모델이 우리 테스트 이미지로 학습했을 수 있다) | train/test 목록을 알 수 있는 모델(A2)을 우선 쓴다. 나머지는 "보고 정확도 확인 불가"로 표기한다. pHash로 near-duplicate를 검사한다. 누수가 의심되면 E8(외부 데이터) 결과를 주 근거로 쓴다 |
| 전처리 불일치 (크기, 정규화, RGB/BGR, 라벨 순서) | 원 코드와 카드에서 전처리를 그대로 옮긴다. 라벨 순서는 `labels.txt`/`config.json`의 id2label로 확인한다. 정확도가 우연 수준이면 전처리부터 의심한다 |
| TF/Keras 모델의 Grad-CAM 레이어 지정 | `model.summary()`로 마지막 Conv 출력을 찾고, `tf.keras.Model(inputs, [conv_out, preds])`를 만들어 `GradientTape`로 계산한다. Teachable Machine은 Sequential 안에 Sequential이 중첩돼 있어 내부 그래프를 functional로 다시 엮어야 한다. 또는 tf2onnx → onnx2torch로 바꿔 pytorch-grad-cam으로 통일한다(변환 후 출력 일치를 검증) |
| ViT/Swin에 Grad-CAM 적용 | pytorch-grad-cam의 `reshape_transform`으로 토큰을 2D로 바꾼다. 결과가 CNN과 직접 비교 가능하지 않다는 점을 명시한다 |
| fastai v1 pickle 호환성과 보안 | 격리 환경에서 한 번만 열어 `state_dict`로 변환해 safetensors로 저장한다. 그 뒤로는 safetensors만 쓴다. 변환 전후 출력 일치를 검증한다 |
| 앙상블 모델 | 개별 멤버 CAM과 평균 CAM을 모두 보고한다. 멤버 간 불일치는 E4 근거로 쓴다 |
| 라이선스 | kmader 데이터가 CC BY-NC-ND일 수 있으므로 비상업 연구 용도로만 쓰고, 데이터를 재배포하지 않는다 |

### 가중치 파일 보안 규칙
- 모든 가중치는 `models/raw/<name>/`에 두고 신뢰하지 않는다. 받자마자 SHA256을 `models/CHECKSUMS.txt`에 기록한다.
- `.safetensors` → `safetensors.torch.load_file`.
- `.pth/.pt` → `torch.load(..., map_location="cpu", weights_only=True)`. 실패해도 `weights_only=False`로 바꾸지 않고 원인을 보고한다.
- `.pkl/.pickle/.bin(pickle)` → **위험하다(임의 코드 실행 가능).** `pickletools`로 opcode를 먼저 점검하고, 네트워크를 끊은 일회용 환경에서만 연다.
- `.h5` → `h5py`로 `model_config`를 먼저 열어 Lambda·custom 레이어가 있는지 검사한 뒤 로드한다. `.keras`는 `safe_mode=True`.
- 저장소 안의 스크립트(`download_weights.py` 등)는 내용을 읽은 뒤에만 실행한다. 데이터 폴더 안의 스크립트는 실행하지 않는다.

---

## 5. 폴더 구조와 환경
```
XAIParkinson/
  PLAN.md  NOTES.md
  data/raw/<dataset>/        # 원본 (신뢰 안 함, 내부 스크립트 실행 금지)
  data/processed/            # 전처리 결과, split 목록 CSV
  models/raw/<name>/         # 받은 가중치 원본 (신뢰 안 함)
  models/safe/<name>/        # 검증·변환한 safetensors
  models/CHECKSUMS.txt
  src/xaipd/
    loaders/                 # 모델별 로더 (keras_tm.py, fastai_resnet.py, hf_swin.py, dat_siamese.py)
    data.py                  # 데이터셋, 전처리 (모델별 전처리 설정)
    cam.py                   # Grad-CAM/++/Score-CAM 래퍼
    metrics.py               # 배경 비율, IoU, pointing game, deletion/insertion, 순위상관
    sanity.py                # 모델 무작위화
  scripts/                   # 01_eval_accuracy.py, 02_cam.py, 03_metrics.py ...
  configs/                   # 모델·데이터 YAML
  results/<exp>/             # 표, 그림, 로그
  envs/                      # requirements 파일
```

**환경 제안** (아직 설치하지 않음)
- 메인: **Python 3.11 venv(또는 uv)**. 이유는 다음과 같다.
  - torch는 3.13 Windows CUDA wheel(2.6 이상)도 있지만, TensorFlow/Keras와 일부 의료영상 라이브러리(MONAI, SimpleITK)는 3.11에서 가장 안정적이다 [추정].
  - DaT 1위 코드는 3.12에서 테스트됐으므로 그 코드만 별도 3.12 uv 환경으로 돌려도 된다.
  - 패키지: `torch torchvision` (cu126 이상 인덱스), `grad-cam` (PyPI 이름. MIT, py>=3.8), `safetensors`, `transformers` (Swin), `timm`, `opencv-python-headless`, `scikit-image`, `scipy`, `pandas`, `matplotlib`, `nibabel`, `SimpleITK`, `imagehash`.
- Keras 모델: 같은 3.11 환경에 `tensorflow-cpu`를 설치한다(Windows 네이티브 GPU는 미지원이고, 추론만 하므로 CPU로 충분하다). 충돌이 나면 `envs/tf` 환경을 분리한다.
- fastai v1 pickle 변환 전용: 일회용 격리 환경(WSL2/Docker, 네트워크 차단, Python 3.7–3.8, fastai 1.0.x). 변환이 끝나면 삭제한다.
- 디스크: 타깃 1은 1GB 미만이고 타깃 2는 가중치 약 0.8GB와 데이터 수 GB다. 여유 229GB로 충분하다.

---

## 6. 단계별 체크리스트와 마일스톤
### M0. 계획 (완료)
- [x] 문헌·가중치·데이터 조사, PLAN/NOTES 작성

### M1. 타깃 1 준비 (사용자 승인 후, 예상 0.5–1일)
- [ ] Python 3.11 환경 생성, 패키지 설치 (약 3–5GB, 10–20분)
- [ ] sayakpaul 저장소 데이터 받기 (data/raw/kmader_pyimagesearch, 약 20MB) + 원본 Kaggle과 대조 (선택)
- [ ] 가중치 받기: Parkonix h5 (작음, MB 단위로 추정), gianlab safetensors 110MB, sayakpaul pkl 43.8MB → SHA256 기록
- [ ] 보안 점검: h5 Lambda 검사, pkl opcode 검사 → 격리 환경에서 state_dict로 변환
- [ ] 노트북과 카드에서 전처리와 라벨 순서 확정

### M2. 타깃 1 정확도 확인, E0 (예상 0.5일)
- [ ] 3개 모델의 test 60장 acc/AUC + bootstrap CI, 보고값과 비교
- [ ] near-duplicate/누수 검사, 1-NN baseline

### M3. 타깃 1 XAI 분석, E1–E8 (예상 2–3일)
- [ ] Grad-CAM/++/Score-CAM 그림
- [ ] 획 마스크 생성 → 배경 비율, pointing game
- [ ] 모델 간 일관성
- [ ] 모델 무작위화 sanity check
- [ ] deletion/insertion
- [ ] 배경·획 마스킹 shortcut 테스트
- [ ] NewHandPD/HandPD zero-shot (데이터 수십 MB)
- [ ] 결과 요약 표와 그림 → `results/target1/`

### M4. 타깃 2 DaTscan (데이터 승인 후, 예상 3–5일)
- [ ] (사용자) DrivenData 데이터 접근 확인 또는 PPMI DUA 승인
- [ ] DaT 1위 저장소 코드 검토 → 가중치 788MB 받기 → weights_only 로드
- [ ] 2·3위 가중치 위치 확인, 3위 전체 뇌 투영 모델 우선 검토
- [ ] 추론 재현 (로그 손실/AUC) → 선조체 아틀라스 정합 → E2/E3/E4/E6
- [ ] PPMI와 대회 데이터 중복 검사

### M5. 정리
- [ ] 주장별 근거 매핑 표 (주장 → 실험 → 수치 → 그림)
- [ ] 한계 기록 (비공식 가중치, 작은 테스트셋, 누수 가능성)

### (대안) M3b. 최소 fine-tuning: M1–M2가 실패할 때만
- [ ] NewHandPD 나선, ImageNet ResNet18/VGG16, 시드 3개, 각 15분 이내
- [ ] 라벨 무작위화 sanity check(E5b)도 이 경로에서만 한다

---

## 7. 결정이 필요한 질문
1. 타깃 1의 가중치들은 "논문 공식 SOTA"가 아니라 공개 구현이다. 이것을 "SOTA"라고 불러도 되는가? 아니면 "공개 고정확도 모델"로 표현할 것인가?
2. pickle 가중치(A2, 같은 테스트셋을 알 수 있는 유일한 모델)를 격리 환경에서 열어도 되는가? WSL2나 Docker를 쓸 수 있는가?
3. DaTscan을 위해 DrivenData 계정 확인이나 PPMI DUA 신청을 진행할 것인가? 신청 명의(개인/학교/지도교수)는 누구로 할 것인가?
4. 메인 환경을 Python 3.11로 따로 두는 데 동의하는가? (시스템 3.13은 건드리지 않는다)
5. Swin(비CNN)을 대조군으로 넣을 것인가?

## 7b. M1 결과 (2026-11-07, 타깃 1, 학습 없음)
환경: `.venv` (Python 3.11.16, uv)
- torch 2.14.1+cu126 (RTX 4070 인식), tensorflow 2.15.1 / keras 2.15 (CPU), grad-cam 1.5.7, transformers 5.22.2, numpy 1.26.4
- 고정 목록: `envs/requirements-lock.txt`

**E0: 같은 테스트셋 60장 (spiral 30 / wave 30)** → `results/target1/e0_accuracy.json`
| 모델 | spiral acc [95% CI] / AUC | wave acc / AUC | 보고값 |
|---|---|---|---|
| A1 MobileNet (Teachable Machine, h5) | 0.80 [0.67–0.93] / 0.90 | 0.57 / 0.64 (spec 0.13, 거의 전부 PD로 예측) | 83% 주장 → 전체 0.68 |
| A3 Swin-Tiny (safetensors, 나선만 학습) | 0.83 [0.70–0.97] / 0.91 | 0.50 / 0.92 (임계값 문제, 전부 PD) | 0.909 (11장) → CI 안이라 재현으로 본다 |
| A2 ResNet-34 fastai (.pkl) | 실행 안 함: unpickle 금지, 격리 환경 미승인 | | |

**E2: 획(stroke) 주목도** → `e2_stroke_attention.json`, 대조군은 `e2_oracle_ceiling.json`
- enrichment = (CAM 에너지 중 획 위 비율) / (획 면적 비율). 무작위 지도는 1.0이다.
- "oracle"은 획 마스크를 같은 7×7 해상도로 만든 지도이며, 이 해상도에서 얻을 수 있는 상한이다.
- spiral
  - A1 Grad-CAM 1.04, A3 1.13 (무작위 1.00, oracle 1.23)
  - pointing game 0.20/0.27 (무작위 0.10–0.17, oracle 0.33)
- wave
  - A1 1.14/1.10/1.09 (Grad-CAM/++/Score-CAM). oracle 1.69, 무작위 1.00이라 무작위 쪽에 가깝다.
  - pointing game 0.13–0.27 (oracle 0.43)
- A3 Score-CAM은 0.79(spiral)와 0.75(wave)로 **무작위보다 낮다.** 획을 피해 간다.
- 정성 관찰 (그림: `results/target1/figures/A1_vs_A3_gradcam_spiral.png`): A1이 그림이 없는 **종이 모서리나 가장자리**를 강하게 강조했다. V01HE01, V04HE01, V07HE01, V01PE01이 그 예다.
- 주의: 7×7 CAM은 해상도가 낮아 spiral에서는 상한 자체가 낮다(1.23). 그래서 spiral보다 wave 결과가 더 뚜렷한 근거다.

**E4: 일관성 (Spearman, top-20% IoU)** → `e4_consistency.json`
- 모델 간 (같은 방법)
  - A1과 A3 Grad-CAM: spiral ρ=0.33, IoU 0.19 / wave ρ=0.09, IoU 0.11
  - Score-CAM: ρ=−0.29 (spiral)
- 같은 모델 안에서 방법 간
  - A3의 Grad-CAM과 Score-CAM: ρ=−0.54 (spiral), −0.56 (wave), IoU 약 0.06 → 정반대다.
  - A1의 Grad-CAM과 Score-CAM: ρ=0.37–0.44

**E7 (간이 shortcut 검사)**: 배경을 흰색으로 지우고 획만 남긴 입력의 정확도
- A1 spiral 0.80 → 0.77, wave 0.57 → 0.50
- A3 spiral 0.83 → 0.80
- 결론을 내리기 이르다(변화가 작다). 배경만 남긴 입력 검사는 아직 하지 않았다.

### M1 체크리스트 갱신
- [x] 3.11 환경, 데이터(두 사본 SHA256 일치), 가중치 SHA256 (`models/CHECKSUMS.txt`)
- [x] A1 h5 검사: Lambda나 커스텀 레이어 없음 (`scripts/inspect_h5.py`) → 로드함
- [x] E0, Grad-CAM/++/Score-CAM 샘플, E2, E4
- [ ] E5 모델 무작위화 sanity check, E6 deletion/insertion, E7 배경만 남긴 입력, E8 NewHandPD zero-shot
- [ ] A2: 격리 환경 승인을 기다리는 중

## 7c. 손그림 E5/E6 (2026-11-07) → `results/target1/e5_e6.json`, `e6_a1_followup.json`
**E5 (계단식 가중치 순열 무작위화 후 원래 Grad-CAM과의 |Spearman|)**
- A1: head 0.32 → top 25% 0.18 → top 50% 0.27 → 전체 0.26
  - 전체를 섞어도 상관이 남는다 → 부분 실패
- A3: head 0.29 → stage4 0.23 → stage3-4 0.17 → 전체 **정의 불가**
  - 무작위 모델의 Grad-CAM이 모든 이미지에서 0인 상수 지도라 Spearman을 계산할 수 없다(재현해서 확인했다).

**E6 (확률 AUC, CAM 순서 vs 무작위 순서)**
- A1 spiral: 삭제 0.44 vs 0.59 → CAM이 충실하다.
- A1 wave: 0.90 vs 0.89 → 차이가 없다.
  - 예측이 포화돼 있다(특이도 0.13, p_max 0.96). 그래서 "이 검사로는 설명을 구분할 수 없다"는 해석도 가능하다.
  - 그러나 log-odds 기준으로 다시 봐도 CAM 4.64 vs 무작위 4.22로 CAM이 더 나쁘다. 정답 이미지만 봐도 4.49 vs 4.03이다.
  - 흰색 baseline이 획을 PD 특징으로 바꿨을 가능성은 남는다.
- A3 wave: 엇갈린다. 삭제는 CAM이 낫고(0.66 vs 0.74), 삽입은 CAM이 나쁘다(0.65 vs 0.75).

**A3 Score-CAM 대상 레이어 재확인**
- 대상 레이어를 `layernorm`에서 마지막 블록 `layernorm_before`로 바꿨다.
- Score-CAM 획 enrichment: 0.77 → 0.80. 둘 다 무작위(1.0)보다 낮다.
- Grad-CAM과의 Spearman: −0.55 → 0.07
- 결론: 획을 피해 가는 결과는 레이어 선택 탓이 아니다.

## 7d. 보행 "움직임" (2026-11-07) → `results/gait_motion/`
**주 대상: Sabo ST-GCN** (공개 가중치, Detectron 2D, 10 fold, UPDRS-gait 회귀)

E0
- 공개 샘플 8개: 앙상블 acc 0.5. 라벨이 거의 0점뿐이라 의미가 적다(15개 중 8개만 쓴 이유는 NOTES 6e).
- CARE-PD 외부 검증(2953 walks, frontal, 몸통 190): acc 0.41, macro-F1 0.24, Spearman −0.19
  - 시점(frontal/back/side/oblique)을 바꿔도 거의 같다.
  - **몸통 정규화 값 100 → 260에 따라 Spearman이 +0.49 → −0.26으로 단조 감소하고, 평균 예측이 1.0 → −0.15로 떨어진다** (`figures/sabo_scale_sensitivity.png`).
  - → 이 수치는 "모델 성능"이 아니라 "입력 스케일 취약성"의 근거다.

**XAI (CARE-PD 60 walks)**
- Grad-CAM은 팔(팔꿈치·손목)에 92%, 다리에 3%를 준다.
- IG와 occlusion은 다리에 43–60%, 팔에 7%를 준다.
- Grad-CAM과 perturbation 방법의 상관은 −0.30 / −0.36이다. IG와 occlusion끼리는 0.78로 일치한다.
- **그룹 occlusion**(관절을 mid-hip으로 대체)
  - 다리를 가리면 출력이 1.14 바뀌고 등급의 94%가 바뀐다.
  - 팔을 가리면 0.53, 54%다.
  - → 모델은 실제로 다리에 더 의존한다.
- 팔 흔들림 감소는 실제 PD 징후다. 하지만 **Grad-CAM의 "팔" 설명은 모델의 실제 의존과 맞지 않는다.** IG도 코(머리)에 25%(균등 분포 기준 7.7%)를 줘서 해석이 어렵다.
- **패딩 shortcut**
  - 공개 샘플은 창의 64%가 앞쪽 0 패딩이다. 근거가 패딩에 놓인 비율은 Grad-CAM 34%, IG 45%, occlusion 55%다.
  - 같은 보행을 0 패딩 없이 넣거나 다른 방식으로 채우면 평균 예측이 0.3–0.4 낮아지고, 등급이 14–18% 바뀐다(CARE-PD 986개).
  - 정답이 0점인 샘플 2개는 0 패딩일 때만 10개 fold 중 9개가 1점으로 판정했다.
- fold 간 Grad-CAM 일치도: 0.54 (공개 샘플은 0.36)
- E5: head만 섞어도 Grad-CAM 0.56, IG 0.71이 남는다. 전체를 섞으면 0.21 / 0.43이다. IG는 sanity check에서 크게 실패했다.
- E6 (출력을 [baseline, 원래 출력]으로 정규화한 AUC)
  - Grad-CAM: 삭제 0.42 vs 무작위 0.28, 삽입 −0.09 vs 0.28 → **무작위보다 나쁘다.**
  - IG: 삭제 −0.45 vs 0.28 → 충실하다.

**보조: CARE-PD joint×time CNN** (재현 학습 모델): **실행하지 못했다.**
- `.gpu_busy`가 사라지길 기다리는 동안, 시스템 메모리 부족으로 Claude Code가 백그라운드 셸을 정리하면서 이 작업도 중단됐다. 학습은 시작 전이었다.
- 지시에 따라 자동으로 다시 시작하지 않았다. 사용자 승인이 필요하다.
- 스크립트: `scripts/carepd_cnn.py` (GPU, `.gpu_busy` 대기, 6GB 가드)
- MotionBERT 고정 인코더 + linear probe도 아직 하지 않았다. CARE-PD 체크포인트(Google Drive)를 받아야 하고, SMPL에서 h36m 관절로 바꾸는 단계가 필요하다.

**모달리티 비교 표**: `results/cross_modality/cross_modality_table.md` (.csv), `common_metrics_bars.png` (CNN 행은 빠져 있다)

## 8. 진행 로그
- 2026-11-07: 1차 계획 작성 → 사용자 지시("학습하지 말고 공개 가중치로 추론 + Grad-CAM")에 따라 공개 가중치 중심으로 다시 작성했다.
- 2026-11-07: M1 실행. 결과는 7b절. 명칭은 "공개 사전학습 모델"로 통일한다(SOTA라는 표현은 보류).

## 7e. 발표용 스켈레톤 애니메이션 (2026-11-07)
- 위치: `results/gait_motion/animations/` (GIF·MP4 3개씩 + `skeleton_montage.png`)
- 구성: Grad-CAM | IG | Occlusion 세 패널, 10-fold 평균 attribution
- 핵심 메시지: Grad-CAM은 팔을 칠하지만, occlusion으로 확인한 실제 의존은 다리다. 정답 0점 샘플 2개는 근거의 34–79%가 0 패딩 프레임에 놓인 채 1점으로 오판됐다.

## 9. 해석 가능 보행 모델 PoC (2026-11-08)
- 목표: 증상 연산자 c_k, 단조 함수 g_k, 벌점 잔차 r를 더하는 모델로 "설계에 의한 해석 가능성"을 보이는 PoC. 식과 파일은 RESEARCH_CONTEXT.md 4절에 있다.
- 데이터: CARE-PD 4개 코호트, 2953 walks, 등급 0/1/2+, 피험자 단위 5-fold(`results/poc/splits.json`)
- 결과 (`results/poc/poc_results.md`)
  - macro-F1: 제안 0.61, 로지스틱 0.66, RF 0.64, 블랙박스 CNN 0.58
  - 잔차 비율 2%
  - 패딩·크기 변화에 대한 등급 변화: 제안 0, CNN 0.23–0.39
  - 주입 검사에서 제안 모델 설명의 적중률: 팔 1.00, 보폭 1.00, 비대칭 0.58. CNN Grad-CAM은 0.29 / 0.60 / 0.17이다.
- 정직하게 밝힐 한계
  - 정확도는 로지스틱보다 낮다.
  - 연산자 파라미터 학습으로 얻은 이득이 없다.
  - 주입 검사의 적중률은 부분적으로 구조상 보장된 것이다.
- 산출물
  - 논문 `paper/main.pdf` (4쪽)
  - 데모 `results/poc/demo/poc_demo.mp4` (36초, 0.5MB) + 미리보기 GIF
  - 문헌 `docs/literature_check.md`
  - 저장소 문서: README / RESEARCH_CONTEXT / AGENTS, `.gitignore`, `results/repo_audit.md`
- [x] GPU 창 03:00–03:04: CNN 시드 1·2(macro-F1 0.54, 0.54), ST-GCN 블랙박스(0.60, 1.25배 크기 변화에 등급 80% 변화, Grad-CAM 적중 0.61/0.76/0.68이지만 비대칭에서는 등급이 6%만 오르는데 지도는 68% 이동). 시드 간 attribution 일치도는 Grad-CAM 0.41, IG 0.75. 논문과 표에 반영했다. `.gpu_parkinson_done`을 기록했다.
