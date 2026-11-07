# XAIParkinson — 조사 노트 (원자료)

작성: 2026-11-07. 계획 수립 단계. 다운로드·설치·학습은 하지 않았다.
표기: [확인] 페이지에서 직접 확인함 / [추정] 추측이거나 2차 출처 / [미확인] 확인 못 함.

---

## 0. 방향 (2026-11-07 사용자 지시)
- 직접 학습하지 않는다. **공개된 사전학습 체크포인트를 받아 추론과 Grad-CAM만** 돌리는 것이 기본 경로다.
- 선정 기준 1순위는 "학습된 체크포인트가 공개돼 있는가"이고, 2순위는 "그 모델을 학습시킨 공개 데이터(같은 테스트셋)가 있는가"다.
- 가중치가 없으면 짧은 fine-tuning을 대안으로 두되, 최소한으로만 쓴다.
- 가중치 파일은 신뢰하지 않는 데이터로 다룬다(아래 6절).

---

## 1. 공개 가중치 조사 결과 (핵심)

### 1.1 손글씨: 나선·물결 그리기 (Kaggle kmader / PyImageSearch, Zham et al. 2017 원본)
| 후보 | 링크 | 형식·프레임워크 | 모델 | 학습 데이터 | 보고 성능 | 라이선스 | 비고 |
|---|---|---|---|---|---|---|---|
| sayakpaul/Parkinson-s-Disease-Classifier | https://github.com/sayakpaul/Parkinson-s-Disease-Classifier | `models/parkinson-predictor.pkl` (43.8MB), fastai learner **pickle** | ResNet-34 (fastai 태그) | PyImageSearch spiral+wave 데이터. **저장소 안에 train 144장·test 60장이 같이 들어 있다** [확인] | README에는 없다. 노트북 안에 있을 수 있다 [미확인] | 표기 없음 [확인] | 같은 테스트셋으로 정확도를 확인할 수 있다. 다만 pickle이라 위험하고, fastai v1(2019)일 가능성이 높아 환경을 맞추기 어렵다 [추정] |
| SaiJeevanPuchakayala/Parkonix | https://github.com/SaiJeevanPuchakayala/Parkonix | `keras_model.h5` + `labels.txt` (Keras HDF5) | Teachable Machine으로 export. MobileNet 계열 [추정] | spiral/wave 스케치. 정확한 출처 미기재 | "83% accuracy" (자체 주장) | 표기 없음 | h5는 pickle보다 안전한 편이다. Teachable Machine 모델은 중첩 Sequential이라 Grad-CAM용 레이어를 지정하기 까다롭다 |
| gianlab/swin-tiny-…-finetuned-parkinson-classification | https://huggingface.co/gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification | `model.safetensors` (110MB), HF Transformers, PyTorch | Swin-Tiny (**CNN 아님**) | kmader spiral (카드에 링크 있음) | acc 0.9091. 평가셋은 약 11장으로 보인다 [추정: 1/11 배수] | 모델 Apache-2.0. 카드에 데이터는 CC BY-NC-ND 4.0으로 적혀 있다 | 가장 안전한 형식이다. 평가 split이 공개되지 않아 kmader test와 학습 데이터가 겹칠 수 있다(누수 위험) |
| skohli01/finetuned-parkinson-classification | https://huggingface.co/skohli01/finetuned-parkinson-classification | safetensors, Swin-Tiny | Swin-Tiny | "imagefolder" (미기재) | acc 0.9091 | Apache-2.0 | gianlab와 같은 레시피로 보인다 [추정] |
| dhhd255/EfficientNet_ParkinsonsPred | https://huggingface.co/dhhd255/EfficientNet_ParkinsonsPred | HF Transformers (PyTorch) | EfficientNet | kmader | 약 83% (주장) | MIT | **분류 헤드가 저장돼 있지 않다.** 예시 코드가 매번 랜덤 Linear 헤드를 만든다. 사용 불가로 본다 |
| DhruvW/parkinsons-spiral-model | https://huggingface.co/DhruvW/parkinsons-spiral-model | `parkinsons_spirals_model.pth` (136MB), pickle 가능성 | 미상 | 미상 | 미상 | 미상 | 카드 없음. 우선순위 낮음 |
| yuchenlung/parkinsons-cnn | https://huggingface.co/yuchenlung/parkinsons-cnn | `parkinson_disease_probability.h5` (0.97MB) + inference.py | 소형 CNN [추정] | 미상 | 미상 | 미상 | 카드 없음 |
| neelanjan00/Parkinsons-Detection | https://github.com/neelanjan00/Parkinsons-Detection | `spiral_model.pickle`, `wave_model.pickle` | sklearn 계열로 보인다 (HOG+RF) [추정] | — | — | — | CNN이 아니고 pickle이다. 제외 |
| FernandoPCJ/parkinson-drawings-cv | https://github.com/FernandoPCJ/parkinson-drawings-cv | 가중치 없음 (`models/.gitkeep`). ONNX export 스크립트만 있다 | ResNet18 | Kaggle "Parkinson Yolo Dataset" (CornelioAC, CC BY 4.0) | 아래 2절 | MIT | 가중치는 없지만 **누수 감사와 Grad-CAM 분석 결과가 우리 주장과 직접 관련된다** |

### 1.2 DaTscan SPECT
| 후보 | 링크 | 형식 | 모델 | 학습 데이터 | 성능 | 라이선스 | 비고 |
|---|---|---|---|---|---|---|---|
| DrivenData "DaT Parkinson's Challenge" 1위 (The Avengers) | 코드: https://github.com/drivendataorg/dat-parkinsons ("1st Place") / 가중치: https://huggingface.co/dinhviettoanle/dat-parkinson-models | `02-siamese/mashup_ensemble_exp_{0..19}_fold_{0..4}.pth` 100개, 개당 7.5MB, 합계 약 788MB. `01-soft_labeling/` 폴더도 있다 | 선조체를 tight crop한 다중 뷰 2D MIP를 입력으로 받는 Siamese CNN, 100개 모델 평균 | 대회 데이터(SFMN, 여러 장비·기관, NIfTI). **DrivenData 로그인이 필요하고 대회 후 접근 가능 여부는 미확인** | private log loss 0.2532 | 코드 MIT. HF 가중치 Apache-2.0 | 추론 스크립트(`submission_src/main.py`)가 있다. Python 3.12에서 테스트됨. 입력이 선조체 crop이라 "선조체 밖에 주목하는가"는 crop 안에서만 볼 수 있다 |
| 같은 대회 2위 (NavAttack) | 위 저장소 "2nd Place" | `scripts/install_weights.py`, `models/SHA256SUMS.*.txt`. 가중치 위치는 미확인 | 선조체 회귀 모델 + 3D 반구별 MIL CNN + 2D 투영 EfficientNet + TabICLv2 블렌드 | 대회 데이터 | 0.2731 | MIT | SHA256 체크섬이 있어 무결성을 검증할 수 있다 |
| 같은 대회 3위 (paulonium) | 위 저장소 "3rd Place" | 가중치 스크립트 미확인 (트리 일부 잘림) | 2D 투영 맵 DenseNet121, 좌우 sagittal 융합, 50개 앙상블 | 대회 데이터 | 0.2732 | MIT | 전체 뇌 투영이면 선조체 ROI 실험에 가장 알맞다 [미확인] |
| mnaeimaei/parkinson-datscan-classifier | https://github.com/mnaeimaei/parkinson-datscan-classifier | 체크포인트 공개 여부 미확인 (보이는 트리에는 없음) | 3D/2.5D CNN, Swin3D 등 60개 실험. 최고는 R3D-18 | 1,362명 (출처 미기재. 대회 데이터로 보인다 [추정]) | OOF AUROC 0.945 | 미기재 | 전처리 문서가 자세하다(템플릿 정합, 후두엽 정규화, 44×44×36 ROI) |
| IoBT-VISTEC/PPMI_DL (Pianpanit et al., IEEE Sensors J 2021) | https://github.com/IoBT-VISTEC/PPMI_DL | **가중치 없음.** TF/Keras 코드만 있다 | 3D PD-Net 4종 + XAI 6종 (Grad-CAM, SHAP 등) | PPMI | — | 미기재 | 가중치가 없으면 학습해야 한다. 3D XAI 구현 참고용 |
| RichardDelwin/Parkinsons-DaTscan (Magesh et al. 2020 코드) | https://github.com/RichardDelwin/Parkinsons-DaTscan | 가중치 없음 (보이는 범위). **PPMI에서 유래한 PNG 슬라이스가 저장소에 올라가 있다** | VGG16 + LIME | PPMI 642장 | 95.2% | 미기재 | PPMI 데이터를 DUA 없이 재배포한 것일 수 있다. **PPMI DUA를 받기 전에는 쓰지 않는다** |

### 1.3 MRI
- HF에 Keras 모델이 몇 개 있다: `PutraDanish/convnextBase-parkinson-mri`, `hou88hou/NeuroScan-AI-Parkinson-MRI`, `maggisai/parkinsons-detection`, `Kashish-jain/Parkinsons-trained-models`(MIT). **모두 모델 카드가 없어** 학습 데이터, 전처리, 성능을 알 수 없다. 재현성을 확인할 수 없으니 1차 타깃에서 제외한다.
- 논문 기반 공개 가중치는 찾지 못했다. Shinde et al. 2019 (NM-MRI, ResNet50, 테스트 80%, NeuroImage: Clinical, https://pmc.ncbi.nlm.nih.gov/articles/PMC6417260)와 3D CNN T1 연구(bioRxiv 2025, https://www.biorxiv.org/content/10.1101/2025.11.17.688918.full.pdf, 84.5%)는 데이터가 PPMI이거나 비공개다.

### 1.4 음성
- 공개된 CNN 가중치는 찾지 못했다. CustNetGC-App (https://github.com/Abishekkarthik07/CustNetGC-Parkinsons-Prediction-App)은 스펙트로그램 CNN과 Grad-CAM 코드가 있지만 `models/` 폴더에 README만 있다.
- `jakub-02/jb_whisper_parkinson_model`은 Whisper 기반이라 CNN이 아니다.
- Sakar 2019 UCI 데이터는 **추출된 특징 표**라서 스펙트로그램 CNN에 쓸 수 없다.

---

## 2. 우리 주장과 직접 관련된 선행 증거
- **FernandoPCJ/parkinson-drawings-cv** (MIT, 2025–26) [확인]
  - Kaggle Parkinson Yolo Dataset에서 원래 split이 누수돼 있었다. validation 원본의 35%가 train에 거의 같은 쌍을 갖고 있었다.
  - 이미지의 약 90%는 회전·반전 복사본이다. 고유 그림은 나선 345–412장, 물결 175–191장 정도다.
  - 학습하지 않는 1-NN만으로 hold-out AUC 0.94–0.96이 나왔다.
  - ResNet18의 AUC 0.969/0.961은 "설명되지 않은" 결과라고 적었다.
  - Grad-CAM의 획(stroke) 집중도는 나선 1.22, 물결 0.97로, 물결에서는 획에 집중하지 않았다.
  - 피험자 ID가 없어 라벨이 질병이 아니라 사람이나 출처를 반영할 수 있다.
  - 결론: 이 데이터셋에서 높은 정확도는 shortcut 때문일 가능성이 있다.
- **Saporta et al. 2022**, "Benchmarking saliency methods for chest X-ray interpretation", Nature Machine Intelligence 4:867–878, doi:10.1038/s42256-022-00536-x. Grad-CAM을 포함한 7가지 saliency 방법이 모두 사람 전문가보다 위치를 크게 못 맞혔다.
- **Adebayo et al. 2018**, "Sanity Checks for Saliency Maps", NeurIPS. 모델·라벨 무작위화 검사. https://arxiv.org/abs/1810.03292
- **Arun et al. 2021**, "Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging", Radiology: AI [미확인: 이번 검색에서는 찾지 못했다. 서지는 기억에 의존].
- DaTscan 3D CNN Grad-CAM이 선조체 밖 피질·피질하 영역에 주목했다는 포스터가 있다 (EUSSET 2025, https://dl.eusset.eu/handle/20.500.12015/5334). 근거가 약하다(포스터).
- 나선 그리기 Grad-CAM 연구 대부분(ParkXplainer, Frontiers Aging Neurosci 2025, https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2025.1702829/full; TalTech 석사논문 https://digikogu.taltech.ee/en/Download/82a9978f-3114-463f-bf73-e1bc19149d5b)은 히트맵이 "임상적으로 타당하다"고 정성적으로만 주장한다. 정량 검증은 없다. 우리 실험이 이 빈틈을 메운다.

---

## 3. 모달리티별 SOTA (문헌, 가중치 없음 포함)
| 모달리티 | 논문 (연도) | 아키텍처 | 보고 정확도 | 코드 | XAI 사용 |
|---|---|---|---|---|---|
| 나선/물결 (Kaggle) | Exploring the Efficacy of Modified Transfer Learning… (arXiv 2510.05015, 2025) https://arxiv.org/abs/2510.05015 | VGG16(나선)/VGG19(물결) + attention | 나선 F1 약 0.90 | 미확인 | 미확인 |
| 나선/물결 | Chakraborty et al., ICACT 2020, "Parkinson's disease detection from spiral and wave drawings using CNNs: a multistage classifier approach" | CNN 앙상블 | 약 93% [추정: 기억에 의존] | 미확인 | 없음 |
| HandPD/NewHandPD/Parkinson Drawing | IJPRAI 2024, doi:10.1142/S0218001424400019 | VGG16, PD-Detect2 | 95% / 85%, 증강 후 100% | 미확인 | Grad-CAM 사용 |
| NewHandPD | Bensefia et al. 2026, https://accedacris.ulpgc.es/handle/10553/156293 | CNN | 94.74%, 증강 후 100% | 미확인 | 미확인 |
| HandPD 원조 | Pereira et al., SIBGRAPI 2016 / CMPB 2016 | CNN (OPF 비교) | 나선 약 87–89% | 미확인 | 없음 |
| PaHaW | Drotár et al., AIM 2016 | 동역학 특징 + SVM (CNN 아님) | — | — | — |
| DaTscan | Magesh et al. 2020, Comput Biol Med 126 (arXiv 2008.00238) | VGG16 TL | 95.2% (sens 97.5, spec 90.9) | 저장소 있음, 가중치 없음 | LIME |
| DaTscan | Pianpanit et al. 2021, IEEE Sensors J, doi:10.1109/JSEN.2021.3077949 | 3D PD-Net | 미확인 | Keras 코드 | Grad-CAM, SHAP 등 6종 |
| DaTscan | Dehghan et al. 2024 (jemsc.qom.ac.ir/article_3048) | 2D CNN | 98.5% (증강) | 미확인 | Grad-CAM |
| DaTscan | DrivenData DaT Challenge 2026 1–3위 | Siamese CNN / DenseNet121 / 블렌드 | log loss 0.25–0.27 | **코드와 가중치 공개 (MIT)** | 없음 [추정] |
| NM-MRI | Shinde et al. 2019, NeuroImage: Clinical | ResNet50 | 테스트 80% | 미확인 | CAM 계열 |
| T1 MRI | 3D CNN (bioRxiv 2025) | 3D CNN | 84.5% (PPMI 100 대 100) | 미확인 | gradient saliency |
| 음성 (Italian PVS) | Costantini et al. 2023, PMC9962335 | 맞춤형 CNN | 특징 기반 ML과 비슷 | 미확인 | 미확인 |
| 음성 (Italian PVS) | MDPI Information 16(2):135 (2025) | mel+MFCC DL + attention | 99% (검증 방식이 불명확) | 미확인 | 미확인 |

---

## 4. 데이터셋
| 이름 | 모달리티 | 크기 | 접근 | 경로 | 예상 용량 |
|---|---|---|---|---|---|
| Kaggle Parkinson's Drawings (kmader) = PyImageSearch | 나선·물결 PNG | 204장 (각 72 train/30 test), 피험자 55명 | Kaggle 로그인. sayakpaul 저장소에 같은 이미지가 들어 있다(GitHub, 로그인 불필요) | https://www.kaggle.com/datasets/kmader/parkinsons-drawings | 약 20MB [추정] |
| Parkinson Yolo Dataset (CornelioAC) | 나선·물결 (증강 사본이 많음) | 이미지의 약 90%가 증강 사본 | Kaggle 로그인, CC BY 4.0 (게시자 주장) | Kaggle 검색 "Parkinson Yolo Dataset" | 미확인 |
| HandPD | 나선·미로 | 92명, 736장 | 바로 다운로드 (UNESP). 인용 요구, 라이선스 표기 없음 | https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/ | 수십 MB [추정] |
| NewHandPD | 나선·미로·원 + 신호 | 66명 (PD 31/HC 35), 이미지 264장 | 바로 다운로드 | 같은 페이지 | 수십 MB [추정] |
| PaHaW | 태블릿 온라인 필기 | PD 37 / HC 38 | **라이선스 계약서를 이메일로 보내야 한다** (mekyska@feec.vutbr.cz, CC peter.drotar@tuke.sk, irena.rektorova@fnusa.cz) | https://bdalab.utko.feec.vutbr.cz | 작음 |
| PPMI DaTscan | SPECT | 수천 건 | **DUA 신청 후 승인 필요(목표 1주)**, LONI IDA. 일부 DaTscan은 Tier 3 | https://www.ppmi-info.org/access-data-specimens/download-data | 원본 DICOM 수 GB [추정] |
| DaT Challenge 데이터 | SPECT NIfTI | 1,362명 [추정] | DrivenData 로그인 + 대회 약관. 대회 후 접근 가능 여부 미확인 | https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/ | 미확인 |
| NTUA Parkinson | MRI + DaT | 78명, DaT 925건 | 기관 이메일로 요청 | https://sites.google.com/view/dimitrioskollias/databases/parkinsons-dataset | 미확인 |
| Italian PVS | 음성 WAV | 65명, 495 녹음 [arXiv 2111.10207 기준] | IEEE DataPort 무료 계정, CC BY 4.0 | https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech | 565MB |
| Sakar 2019 (UCI) | 음성 **특징 표** | 252명 × 3회 | 바로 다운로드 | UCI ML repo | <10MB |
| mPower | 스마트폰 음성 등 | 약 9,500명 | Synapse 계정 + 윤리 설문 + 조건 동의 (qualified researcher) | Synapse | 큼 |
| Mendeley spiral | 나선 | 미확인 | 바로 다운로드 | https://data.mendeley.com/datasets/fd5wd6wmdj | 미확인 |

---

## 5. 환경 확인
- PyTorch Windows CUDA wheel(cp313-win_amd64)은 cu126 인덱스에 torch 2.6.0부터 있다 [확인: download.pytorch.org/whl/cu126/torch/]. PyPI 기본 인덱스는 Windows에서 CPU 빌드만 제공한다.
- `grad-cam` (jacobgil, PyPI 이름은 `grad-cam`, 최신 1.5.7, MIT): requires_python >=3.8. 의존성은 torch>=1.7.1, torchvision>=0.13, ttach, opencv-python-headless, scikit-learn 등이다. 3.13 classifier는 없지만 순수 Python이라 문제없을 것으로 본다 [추정].
- TensorFlow: Windows 네이티브 GPU 지원은 TF 2.10에서 끝났다 [기억]. Keras 모델은 CPU로 추론하면 된다(작은 모델이라 충분). TF의 Python 3.13 지원 여부는 [미확인].
- fastai v1 `.pkl`: 최신 torch나 Python 3.13에서는 거의 확실히 로드되지 않는다 [추정]. 격리된 별도 환경(구버전)이 필요하다.
- DaT 1위 코드는 uv + Python 3.12.13 (Ubuntu)에서 테스트됐다.

## 6. 가중치 보안 메모
- `.pth/.pt`: `torch.load(path, map_location="cpu", weights_only=True)`. torch 2.6부터 기본값이 True다. 실패하면 **임의로 False로 바꾸지 않는다.** 원인을 기록하고 격리 환경에서만 검토한다.
- `.safetensors`: 안전하다. 1순위 형식이다.
- `.pkl/.pickle` (fastai, sklearn) 및 일부 `.bin` (`training_args.bin`도 pickle이다): **임의 코드가 실행될 수 있다.** 로드하지 않거나, 네트워크를 끊은 격리 환경(WSL/Docker, 비밀정보 없음)에서만 연다. `pickletools.dis`로 opcode를 먼저 점검한다(GLOBAL/REDUCE 대상 확인).
- `.h5` (Keras): 먼저 `h5py`로 `model_config` JSON만 읽어 `Lambda` 레이어가 있는지 확인한다(Lambda는 marshal된 바이트코드를 담을 수 있다). `.keras` 형식은 `safe_mode=True`.
- ONNX: 비교적 안전하다(실행 코드 없음).
- 받은 즉시 SHA256을 기록한다. DaT 2위는 공식 SHA256SUMS를 제공한다.

## 6b. M1 실행 기록 (2026-11-07)
- 데이터 출처
  - `data/raw/parkonix_repo`: SaiJeevanPuchakayala/Parkonix, commit a5859306, sparse `Data/drawings`
  - `data/raw/sayakpaul_repo`: sayakpaul/Parkinson-s-Disease-Classifier, commit 67d3b779
  - 두 사본 모두 204장이고 **SHA256이 전부 일치한다.** 테스트 60장(spiral 15+15, wave 15+15)도 같다. train/test 사이에 완전히 같은 이미지는 없다.
  - 해시 목록: `data/raw/*_sha256.txt`
  - 원 출처는 Kaggle kmader/parkinsons-drawings (Zham et al. 2017)이다. 라이선스는 CC BY-NC-ND 4.0으로 표기된 곳이 있다(gianlab 카드). 비상업 연구로만 쓰고 재배포하지 않는다.
- Parkonix 코드에서 확인한 사실
  - 테스트 경로가 `./Data/spiral/testing/...`이다. A1은 이 train 폴더로 Teachable Machine 학습을 한 것으로 보인다 → **testing 폴더가 같은 테스트셋** [추정, 강함].
  - 전처리: `ImageOps.fit` 224 center-crop (LANCZOS), `x/127.5-1`
  - 라벨: 0 Healthy, 1 Parkinson
- A3 eval 샘플은 11개다(runtime × samples/s로 역산). 학습 72장의 약 15% split으로 보여 kmader test는 학습에 쓰이지 않았을 것이다 [추정].
- 가중치 SHA256: `models/CHECKSUMS.txt`
  - A2 pkl은 `models/raw/a2_sayakpaul_fastai/`에 두었다. **unpickle 금지** (격리 환경 승인 전).
  - A3는 `training_args.bin`(pickle)을 받지 않았다.

- 구현 메모
  - transformers 5.x는 v4 체크포인트 키 이름이 다르다. 그래서 `from_pretrained(use_safetensors=True, local_files_only=True)`로 로드해 키 변환을 적용했다. missing/unexpected 키가 0개인지 assert로 확인한다.
  - A1 Grad-CAM 대상 레이어는 MobileNetV2 `out_relu` (7×7×1280)다. 중첩 Sequential을 functional로 다시 엮었고, 원 모델과 출력이 같은지(1e-4) assert로 확인한다.
  - A3 대상 레이어는 `swin.layernorm`이고, reshape로 7×7로 바꾼다.
  - A1의 Score-CAM은 TF로 직접 구현했다(`src/xaipd/a1_keras.py`).
- Swin Score-CAM이 무작위보다 낮은 이유가 구현 탓(layernorm 토큰 → 마스크)일 수 있다. 다른 대상 레이어(마지막 블록 `layernorm_before`)로 다시 확인할 필요가 있다.

## 6c. 보행(gait) 확장 기록 (2026-11-07, 진행 중)
- **데이터: PhysioNet gaitpdb 1.0.0**
  - 라이선스 ODC-By 1.0, 로그인 없음
  - PD 93명 / HC 73명, VGRF 16센서 + 좌우 합계, 100Hz
  - PhysioNet zip 다운로드는 약 20KB/s로 너무 느렸다. 그래서 S3 미러(physionet-open)에서 파일 단위로 병렬 다운로드하고, 공식 `SHA256SUMS.txt`로 검증하는 중이다.
- **G1 (공개 가중치)**: Naimi et al. 2023, "1D-Convolutional transformer for PD diagnosis from gait" (arXiv 2311.03177)
  - 저장소: https://github.com/SafwenNaimi/1D-Convolutional-transformer-for-Parkinson-disease-diagnosis-from-gait
  - 가중치 `Pretrained Model/main.hdf5`, SHA256 163bb3e0…
  - h5 검사 결과 Lambda는 없다. TFOpLambda 56개는 모두 `__operators__.add`, `math.add`, `expand_dims` 같은 TF 기본 연산이다.
  - **실제 구조는 Conv 레이어가 없는 transformer-only 모델이고, 출력은 4클래스(중증도) softmax다.** 그래서 Grad-CAM은 쓸 수 없고 IG와 occlusion만 쓴다. P(PD) = 1 − p(class0)로 계산한다.
  - 어떤 fold로 학습했는지 알 수 없다. 같은 166명 데이터로 학습했을 것이므로 정확도에는 **누수 가능성이 있다.**
- **G2 (재현 학습 모델, 공개 가중치 아님)**: El Maachi et al. 2020 (arXiv 1910.11509) 18-branch parallel 1D-CNN을 PyTorch로 옮겼다(`src/xaipd/gait.py`).
  - 원 코드: https://github.com/imanneelmaachi/Parkinson-disease-detection-and-severity-prediction-from-gait (가중치 없음)
  - 전처리: 100샘플 구간, step 50, 구간별 L2 정규화. 원 코드의 `norm == 'std '` 오타 때문에 실제로 돌아가는 것은 L2다.
  - 학습: subject 단위로 25%를 hold-out, 시드 3개, CPU 전용
- 그 밖의 확인: WCSTNet (SubhadeepHyperX) 저장소에는 그림만 있고 가중치와 코드가 없다.
- **사고 기록**: `images_e5_e6.py`가 메모리를 47GB 커밋했다(RAM은 31GB).
  - 원인: A3 Score-CAM에 60장을 한 번에 넣었다. 60 × 768채널 × 224² 마스크 × 3채널이 수십 GB가 된다.
  - 수정: 이미지를 한 장씩 처리하고 batch_size를 16으로 줄였다. `src/xaipd/memguard.py`를 추가했다(RSS가 6GB를 넘으면 스스로 종료, 스레드 4개로 제한). gait 스크립트에도 적용했다.
  - 아직 할 일: psutil 설치, 이미지 5장으로 메모리 smoke test, 그다음 본 실행
  - 현재 상태: 사용자 결정을 기다리는 중이라 실행을 멈췄다. 이전 프로세스(PID 25852)는 지시에 따라 건드리지 않았다.

## 6d. 보행 "움직임" 모달리티 (사용자 지시로 전환. gaitpdb는 보류하고 받은 파일은 보존)
- gaitpdb: 306개 txt를 S3에서 받았다. 1044초 걸렸고 SHA256SUMS로 검증했다. 보존만 하고 분석은 보류한다.
- **주 대상: Sabo et al. 2021**, "Estimating Parkinsonism Severity in Natural Gait Videos of Older Adults with Dementia" (arXiv 2105.03464)
  - 저장소: https://github.com/TaatiTeam/stgcn_parkinsonism_prediction
  - **라이선스(UHN): 학술·비상업 연구 목적으로만 사용·수정·재배포를 허용하고, 상업 목적은 금지한다.**
  - 공개 가중치: `.pt` 64개. Detectron/AlphaPose/Kinect2D는 각 10 fold × (final, pretrain)이고, Kinect3D는 2개 버전이다. 입력은 COCO 17관절 2D이고 중심을 (100, 100)으로 맞춘다. UPDRS-gait를 회귀한다.
  - 공개 데이터: 라벨(UPDRS_gait, SAS_gait)이 붙은 샘플 보행 15개(csv)뿐이다. 원 데이터 4787 walks는 비공개다. → **E0는 샘플 15개로만 볼 수 있다(한계).**
- **보조: CARE-PD** (NeurIPS 2025 D&B, arXiv 2510.04312)
  - 출처: Dataverse doi:10.5683/SP3/TWIKMK, HF vida-adl/CARE-PD. 둘 다 gated가 아니다.
  - **라이선스: 메타데이터에는 CC BY-NC-ND 4.0, 본문 약관에는 CC BY-NC 4.0으로 서로 다르다.** 비상업 연구로만 쓰고, 재배포와 개인 식별 시도는 하지 않는다. 데이터와 원 코호트 논문을 인용한다.
  - 데이터 형식은 pickle이다 → `src/xaipd/safe_pickle.py`로 opcode를 먼저 검사한 뒤 allowlist unpickler로 연다.
  - 공개 체크포인트는 원래 과제용으로 사전학습된 인코더(MotionBERT 등)뿐이다. PD 분류 헤드는 공개돼 있지 않다 → 재현 학습한다(1D-CNN, 그리고 고정 인코더 + linear probe).
- 제외한 후보
  - Kaggle Li 2018 (앉아서 하는 과제라 보행이 아니고, PD 모델이 없다)
  - Lu 2020 OF-DDNet과 GaitForeMer (데이터 비공개, PD 가중치 없음)
  - PD4T (신청서 필요, 얼굴이 담긴 비디오)
  - TULIP-Bench (접근 제한)
  - PD-gait-VSDF (가중치나 데이터 확인 불가)

### 6e. 보행 실행 기록 (2026-11-07)
- **공개 샘플 15개 중 8개만 쓴 이유**
  - 15개의 구성: detectron 4, alphapose 4, kinect2d 4, kinect3d 3. 15개 모두 UPDRS_gait 라벨이 있다.
  - 각 샘플은 같은 검출기로 학습한 모델에만 넣을 수 있다.
  - 이번에 받은 가중치는 detectron(10 fold)과 kinect2d(10 fold)뿐이다. alphapose 가중치는 330MB라 받지 않았다. kinect3d는 3D 입력에 gait feature 옵션이 붙은 별도 구조라 아직 포팅하지 않았다.
  - 그래서 E0는 detectron 4개 + kinect2d 4개 = 8개다. 라벨 분포는 detectron 4개가 모두 0점, kinect2d가 0점 2개와 1점 2개다. 정확도 0.5는 의미가 거의 없다.
  - 이 샘플들은 Sabo 원 데이터에서 나온 것이므로 해당 모델의 학습 fold에 들어 있었을 가능성이 높다(누수).
- **Sabo 체크포인트**: `torch.load(weights_only=True)`로 모두 열렸고, strict 키 매칭과 `head == classification_head`를 확인했다. SHA256은 `models/raw/sabo_CHECKSUMS.txt`(44개)에 있다.
- **원 코드와 다르게 한 점**
  - 원 평가는 120프레임 random crop이다. 여기서는 stride 15 창을 모두 평균했고, 외부 검증에서는 창을 최대 8개만 썼다.
  - 원 코드의 정수 변환 방식을 찾지 못해 round 후 [0, 2]로 clip했다(추정).
- **CARE-PD 3D → 2D 투영과 정규화** (외부 검증 해석용)
  - SMPL FK(betas=0)로 COCO-13 관절을 만든다. 코는 SMPL vertex 332를 머리 관절에 강체로 붙였다. SMPL 엉덩이 관절은 COCO 엉덩이보다 안쪽에 있다.
  - 정사영(orthographic) 4가지 시점: frontal(u=+x), back, side, oblique45
  - 매 프레임 mid-hip을 (100, 100)에 두고, 보행마다 몸통 길이 중앙값을 190 단위로 맞췄다. 190은 Sabo detectron 샘플 범위 153–217의 중앙값이다.
  - conf=1로 두었고, 검출기 노이즈와 원근 효과는 재현하지 않았다.
- **실행 시간과 메모리**
  - sabo_xai 외부 검증(2953 walks × 6설정 × 10 fold, CPU): 2551초
  - XAI n=120은 memguard가 RSS 6.2GB에서 중단시켰다 → n=60으로 다시 돌렸다.
  - images_e5_e6(구버전 코드, 60장 일괄 Score-CAM): 4059초, 커밋 메모리 최대 약 51GB
  - e6_a1_followup: 약 2분, RSS 0.7GB
- **E6 A1 물결 후속 검사** (사용자 지적 반영)
  - 확률 기준으로는 포화(p_max 0.96) 때문에 둔감하다는 해석이 가능하다.
  - 그러나 log-odds 기준으로 다시 봐도 CAM 순서 AUC가 4.64로 무작위 4.22보다 나쁘다. 정답 이미지 17장만 봐도 4.49 대 4.03이다.
  - 따라서 포화만으로는 설명되지 않는다. 다만 지울 때 흰색으로 채우는 방식(baseline)이 물결 획을 "PD 쪽 특징"으로 바꿨을 가능성은 남는다.

## 7. 불확실한 점 / TODO
- [ ] sayakpaul 노트북의 fastai 버전과 보고 정확도 (노트북 열람)
- [ ] Parkonix의 학습 데이터 출처와 split (kmader test와 겹치는지)
- [ ] gianlab Swin의 eval split (`runs/` 텐서보드 로그, `training_args.bin`은 pickle이라 열지 않는다)
- [ ] DaT 대회 데이터를 대회 후에도 받을 수 있는지, 데이터 출처와 라이선스
- [ ] DaT 2위·3위 가중치의 실제 위치와 크기
- [ ] DaT 대회 데이터와 PPMI가 겹치는지 (겹치면 PPMI 평가는 누수)
- [ ] kmader 데이터셋의 정식 라이선스 (gianlab 카드는 CC BY-NC-ND 4.0으로 표기)
- [ ] Chakraborty 2020 수치와 Arun 2021 서지 재확인

### 6f. 중단 기록 (2026-11-07)
- `carepd_cnn.py`(GPU 대기 중)와 대기용 셸 몇 개가 시스템 메모리 부족 때문에 Claude Code에 의해 정리됐다.
  - 우리 스크립트의 메모리 문제는 아니다. sabo_xai의 RSS 최대값은 5.64GB였고 가드 안이었다.
  - 학습 시작 전이라 남은 파일은 없다.
  - 다시 시작하려면 사용자 승인이 필요하다.
- Sabo n=120 XAI는 memguard가 6.2GB에서 스스로 중단시켰다 → n=60으로 완료했다.

### 6g. 스켈레톤 애니메이션 (2026-11-07)
- 스크립트: `scripts/sabo_skeleton_anim.py` (Agg 전용, CPU 4스레드, 6GB 가드). 실행 86초, RSS 최대 1.0GB.
- attribution은 저장해 둔 값을 쓰지 않고 선택한 3개 walk에 대해 **10개 fold 모두로 다시 계산해 평균**했다.
  - 각 fold의 지도를 합이 1이 되도록 정규화한 뒤 평균했다.
  - 기존 npz는 IG와 occlusion이 fold 0 하나뿐이었다.
- 예측은 같은 120프레임 창에 대한 10개 fold 앙상블이다(round 후 [0, 2]로 clip).
- 대상
  - ID_37: GT 0 → pred 1, padding 88/120
  - ID_48: GT 0 → pred 1, padding 66/120
  - CARE-PD 3DGait vid0078_0063b: GT 3 → pred 0, frontal 투영, 패딩 없음
- 결과물: `results/gait_motion/animations/` 아래 GIF 3개, MP4 3개(ffmpeg 8.0.1), `skeleton_montage.png`, `animation_meta.json`
- 관찰
  - Grad-CAM은 팔에 64–88%를 준다. occlusion은 다리에 36–63%를 준다.
  - IG는 머리와 어깨 쪽에 크게 몰린다.
  - 패딩 구간이 받는 비율: Grad-CAM 37–47%, IG 34–79%, occlusion 37–58%

### 6h. PoC 실행 기록 (2026-11-08)
- 문헌 조사: 서브에이전트가 웹 검색으로 했다 → `docs/literature_check.md`. 같은 조합의 선행연구는 찾지 못했다(검색은 망라적이지 않다). CARE-PD UPDRS-gait는 0–3이다.
- 환경
  - shap을 설치하자 numpy가 2.4로 올라갔다. TF 2.15와 호환되도록 numpy 1.26.4 + shap 0.46으로 되돌렸다. lock 파일은 갱신했다.
  - memguard에 BelowNormal 우선순위와 `XAIPD_THREADS`를 추가했다.
- 실행 시간 (CPU 2스레드, BelowNormal)
  - 캐시 생성 11초
  - 제안 모델 + 고전 ML: 279초, RSS 1.4GB
  - 절제 실험 3종: 각 약 4분
  - 블랙박스 CNN: 921초, RSS 1.3GB
  - 데모 렌더링: 약 1분
  - 논문: latexmk(MiKTeX)
- 데모 예시 walk는 PD-GaM 062-15-001268_wid00_2(fold 0 테스트)다. "보이기 좋게 고른 예"라고 영상에 표기했다.
- 처음 짠 GPU 큐는 .gpu_busy 유무만 보고 시작했다. GenUI 단계 사이의 빈틈에 끼어들 위험이 있어 다시 짰다.
  - 새 조건: GenUI 로그에 "GPU YIELD to Parkinson"이 나오고 lock이 없을 때 시작한다.
  - 시작하면 직접 lock을 잡는다.
  - 끝날 때 trap으로 lock을 지우고 `.gpu_parkinson_done`을 쓴다.
  - 전체 마감 38분
  - 예전 큐는 내 PID 18608만 종료했다. PID 31388은 GenUI 것으로 보여 건드리지 않았다.
- 논문 수치는 `scripts/poc_report.py`가 `paper/numbers.tex`와 `tab_*.tex`로 자동 생성한다.
  - 주의: bash heredoc을 거치면 `\`가 `\`로 줄어든다. LaTeX 문자열은 raw string으로 쓴다.
