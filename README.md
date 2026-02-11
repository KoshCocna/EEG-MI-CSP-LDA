🧠 MI-CSP-LDA-EEG-Control

Motor Imagery 기반 6-Class EEG BCI 제어 시스템
CSP + LDA Classification Pipeline

1. 📌 Project Overview

본 프로젝트는 Motor Imagery (MI) EEG 신호를 활용하여
다음 6개의 명령어를 분류합니다:

Left  |  Right  |  Up  |  Down  |  ZoomIn  |  ZoomOut


궁극적인 목표는 해당 분류 결과를 이용하여
🦾 로봇팔 제어 시스템으로 확장하는 것입니다.

2. 🧠 Why Motor Imagery?
Feature	P300	Motor Imagery
방식	자극 기반 ERP	자발적 뇌파 제어
반응 특성	수동적	능동적
이벤트 필요 여부	필요	불필요
실시간 제어	제한적	가능

✅ 로봇팔 제어에는 Motor Imagery 방식이 더 적합

3. ⚙️ Hardware Setup
Item	Specification
Device	Laxtha QEEG-64FX
Channels	24
Sampling Rate	250 Hz
Epoch Length	4 seconds
Frequency Band	8–30 Hz (μ / β rhythm)
4. 📍 Channel Layout
ch0 → ch23
FP1, FP2, F3, F4,
C3, C4, FC5, FC6,
O1, O2, F7, F8,
T7, T8, P7, P8,
AFZ, CZ, FZ, PZ,
FPZ, OZ, AF3, AF4

🎯 Motor Cortex 주요 채널
C3, C4, CZ, FC5, FC6, FZ

5. 🔬 Signal Processing Pipeline
Raw EEG
   ↓
Bandpass Filter (8–30 Hz)
   ↓
Common Spatial Pattern (CSP)
   ↓
Log-Variance Feature Extraction
   ↓
Linear Discriminant Analysis (LDA)
   ↓
Confusion Matrix Evaluation

6. 📊 Output Result

실행 후 다음 파일이 자동 생성됩니다:

result/confusion_matrix.png


Accuracy 출력

Classification report 출력

Confusion matrix 시각화 및 PNG 저장

7. 📁 Project Structure
MI-CSP-LDA-EEG-Control/
│
├── generate_fake_mi_epochs.py
├── train_csp_lda_mi.py
│
├── data/
│   ├── left/
│   ├── right/
│   ├── up/
│   ├── down/
│   ├── zoomIn/
│   └── zoomOut/
│
└── result/
    └── confusion_matrix.png

8. ▶️ How to Run
Step 1 — (Optional) Generate Fake MI Data
python generate_fake_mi_epochs.py

Step 2 — Train & Evaluate
python train_csp_lda_mi.py

9. 🔮 Future Extension

Real-time sliding window classification

Online majority voting

ROS integration

Robot arm serial / UDP control

Filter Bank CSP

Riemannian geometry classifier

EEGNet / CNN 기반 확장

10. 🛠 Tech Stack

Python

NumPy

SciPy

scikit-learn

matplotlib

👨‍🔬 Author

Kanye Kim
BCI · EEG Signal Processing · Wireless Communication

