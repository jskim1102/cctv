
### 0. VER
ver 1.0 - PyQT + 폴리곤 + AI 분석


### 1. ENV
```bash
poetry source add pytorch-gpu https://download.pytorch.org/whl/cu121 --priority=explicit
poetry add --source pytorch-gpu torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
poetry add numpy==1.26.4 opencv-python==4.9.0.80
poetry add ultralytics f_yolov8 f_dpi
poetry add pyqt5==5.15.10 pyqt5-qt5==5.15.2
```
