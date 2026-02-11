🧠 MI-CSP-LDA-EEG-Control

Motor Imagery 기반 EEG 신호를 이용한
6-Class BCI 제어 시스템 (CSP + LDA Pipeline)

🚀 Project Overview

본 프로젝트는 Motor Imagery (MI) EEG 신호를 이용하여

Left / Right / Up / Down / ZoomIn / ZoomOut


6개의 명령어를 분류하고,
궁극적으로 로봇팔 제어 시스템으로 확장하기 위한 BCI 연구 프로젝트입니다.

🧠 Why Motor Imagery?
P300	Motor Imagery
자극 기반 ERP	자발적 뇌파 제어
수동적 반응	능동적 제어
이벤트 필요	실시간 제어 가능

👉 로봇팔 제어에는 Motor Imagery가 더 적합

⚙️ Hardware Setup

Device: Laxtha QEEG-64FX

Channels Used: 24

Sampling Rate: 250 Hz

Epoch Length: 4 seconds

Frequency Band: 8–30 Hz (μ / β rhythm)

📍 Channel Layout (ch0 → ch23)
FP1, FP2, F3, F4,
C3, C4, FC5, FC6,
O1, O2, F7, F8,
T7, T8, P7, P8,
AFZ, CZ, FZ, PZ,
FPZ, OZ, AF3, AF4


Motor cortex 핵심 채널:

C3, C4, CZ, FC5, FC6, FZ

🔬 Signal Processing Pipeline
Raw EEG
   ↓
Bandpass Filter (8–30 Hz)
   ↓
CSP (Common Spatial Pattern)
   ↓
Log-Variance Feature
   ↓
LDA Classifier
   ↓
Confusion Matrix

📊 Result Example

Confusion matrix는 자동으로 result/ 폴더에 저장됩니다:

result/confusion_matrix.png

📁 Project Structure
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

▶️ How to Run
1️⃣ Generate Fake MI Data (optional)
python generate_fake_mi_epochs.py

2️⃣ Train & Evaluate
python train_csp_lda_mi.py


실행 결과:

Accuracy 출력

Classification report 출력

Confusion matrix PNG 자동 저장

🦾 Future Extension

Real-time sliding window classification

Online majority voting

ROS integration

Robot arm serial control

Filter Bank CSP

Riemannian Geometry classifier

EEGNet / CNN 기반 딥러닝 확장

🧩 Research Direction

ERD/ERS 기반 feature 강화

Cross-session generalization

Transfer learning

Multi-subject adaptation

🛠 Tech Stack

Python

NumPy

SciPy

scikit-learn

matplotlib

👨‍🔬 Author

Kanye Kim
BCI / EEG Signal Processing / Wireless Communication
