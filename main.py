import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidgetItem, QCheckBox,
    QWidget, QHBoxLayout, QHeaderView, QAbstractItemView,
)
from PyQt5.QtCore import QTimer, Qt, QPoint, QDateTime, QTimeZone
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QPolygon
from PyQt5 import uic

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

form_class, base_class = uic.loadUiType("main_window.ui")


class MainWindow(base_class, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ── ROI 테이블 초기화 ──
        self._init_roi_table()

        # ── FPS 계산용 ──
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # ── 카메라 ──
        self.cap = None
        for idx in range(5):
            for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
                        print(f"카메라 연결 성공: index={idx}, backend={backend}")
                        break
                cap.release()
            if self.cap:
                break

        if not self.cap:
            print("카메라를 찾을 수 없습니다.")
            self.labelCamera.setText("카메라를 찾을 수 없습니다.")
        else:
            self.labelCameraStatus.setText("카메라: 연결됨")
            self.labelCameraStatus.setStyleSheet(
                "QLabel { color: green; font-size: 12px; font-weight: bold; }"
            )

        # ── GPU 정보 ──
        if HAS_TORCH and torch.cuda.is_available():
            self.labelGPU.setText(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.labelGPU.setText("GPU: 사용불가")

        # ── YOLO 모델 ──
        self.model = None
        if HAS_YOLO: 
            try:
                self.model = YOLO('yolo11n.pt')
                self.labelModel.setText("모델: YOLO11n")
                self.labelModel.setStyleSheet(
                    "QLabel { color: green; font-size: 12px; font-weight: bold; }"
                )
                print("YOLO 모델 로드 완료")
            except Exception as e:
                print(f"모델 로드 실패: {e}")
                self.labelModel.setText("모델: 로드실패")
        else:
            self.labelModel.setText("모델: ultralytics 미설치")

        # ── ROI 관련 ──
        self.roi_mode = False
        self.roi_points = []
        self.roi_polygons = []
        self.current_mouse_pos = None
        self.pixmap_offset = (0, 0)
        self.pixmap_size = (0, 0)

        # ── 버튼 연결 ──
        self.btnROI.clicked.connect(self.toggle_roi_mode)
        self.btnDeleteROI.clicked.connect(self.delete_selected_roi)
        self.btnAI.clicked.connect(self.toggle_ai_mode)

        # ── AI 분석 상태 ──
        self.ai_enabled = False

        # ── 카메라 레이블 마우스 이벤트 ──
        self.labelCamera.setMouseTracking(True)
        self.labelCamera.mousePressEvent = self.camera_mouse_press
        self.labelCamera.mouseMoveEvent = self.camera_mouse_move

        # ── 프레임 타이머 (30ms) ──
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # ── 상태바 타이머 (1초) ──
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)
        self.update_status()

    # ──────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────
    def _init_roi_table(self):
        """테이블 컬럼 구조만 설정 (스타일은 .ui 관리)"""
        self.tableROI.setColumnCount(3)
        self.tableROI.setHorizontalHeaderLabels(["선택", "순번", "좌표"])
        self.tableROI.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableROI.setSelectionMode(QAbstractItemView.NoSelection)
        self.tableROI.verticalHeader().setVisible(False)

        header = self.tableROI.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.tableROI.setColumnWidth(0, 40)
        self.tableROI.setColumnWidth(1, 40)

    # ──────────────────────────────────────────────
    # 상태바 업데이트 (1초마다)
    # ──────────────────────────────────────────────
    def update_status(self):
        self.labelFPS.setText(f"FPS: {self.fps:.1f}")

        kst = QTimeZone(b"Asia/Seoul")
        now = QDateTime.currentDateTime().toTimeZone(kst)
        self.labelDateTime.setText(now.toString("yyyy-MM-dd HH:mm:ss"))

    # ──────────────────────────────────────────────
    # 좌표 변환: QLabel 마우스 좌표 → pixmap 좌표
    # ──────────────────────────────────────────────
    def _label_to_pixmap(self, x, y):
        px = x - self.pixmap_offset[0]
        py = y - self.pixmap_offset[1]
        px = max(0, min(px, self.pixmap_size[0] - 1))
        py = max(0, min(py, self.pixmap_size[1] - 1))
        return (px, py)

    # ──────────────────────────────────────────────
    # ROI 모드
    # ──────────────────────────────────────────────
    def toggle_roi_mode(self):
        if not self.roi_mode:
            self.roi_mode = True
            self.roi_points = []
            self.btnROI.setText("ROI 완료")
            self.labelCamera.setCursor(Qt.CrossCursor)
        else:
            self.roi_mode = False
            if len(self.roi_points) >= 3:
                self.roi_polygons.append(list(self.roi_points))
                self.update_roi_table()
                print(f"ROI #{len(self.roi_polygons)} 설정 완료")
            self.roi_points = []
            self.btnROI.setText("ROI 설정")
            self.labelCamera.setCursor(Qt.ArrowCursor)

    def toggle_ai_mode(self):
        if not self.model:
            print("YOLO 모델이 로드되지 않았습니다.")
            return

        self.ai_enabled = not self.ai_enabled
        if self.ai_enabled:
            self.labelAI.setText("AI 분석: 활성화")
            self.labelAI.setStyleSheet(
                "QLabel { color: green; font-size: 12px; font-weight: bold; }"
            )
            print("AI 분석 활성화")
        else:
            self.labelAI.setText("AI 분석: 비활성화")
            self.labelAI.setStyleSheet(
                "QLabel { color: red; font-size: 12px; font-weight: bold; }"
            )
            print("AI 분석 비활성화")

    def camera_mouse_press(self, event):
        if not self.roi_mode:
            return

        if event.button() == Qt.LeftButton:
            px, py = self._label_to_pixmap(event.pos().x(), event.pos().y())
            self.roi_points.append((px, py))

        elif event.button() == Qt.RightButton:
            if len(self.roi_points) >= 3:
                self.roi_polygons.append(list(self.roi_points))
                self.update_roi_table()
                print(f"ROI #{len(self.roi_polygons)} 설정 완료")
            self.roi_points = []
            self.roi_mode = False
            self.btnROI.setText("ROI 설정")
            self.labelCamera.setCursor(Qt.ArrowCursor)

    def camera_mouse_move(self, event):
        if self.roi_mode:
            px, py = self._label_to_pixmap(event.pos().x(), event.pos().y())
            self.current_mouse_pos = (px, py)

    # ──────────────────────────────────────────────
    # ROI 테이블
    # ──────────────────────────────────────────────
    def update_roi_table(self):
        self.tableROI.setRowCount(len(self.roi_polygons))
        for i, polygon in enumerate(self.roi_polygons):
            cb_widget = QWidget()
            cb = QCheckBox()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.tableROI.setCellWidget(i, 0, cb_widget)

            num_item = QTableWidgetItem(str(i + 1))
            num_item.setTextAlignment(Qt.AlignCenter)
            self.tableROI.setItem(i, 1, num_item)

            coords = ", ".join([f"({x},{y})" for x, y in polygon])
            self.tableROI.setItem(i, 2, QTableWidgetItem(coords))

    def delete_selected_roi(self):
        rows_to_delete = []
        for i in range(self.tableROI.rowCount()):
            cb_widget = self.tableROI.cellWidget(i, 0)
            if cb_widget:
                cb = cb_widget.findChild(QCheckBox)
                if cb and cb.isChecked():
                    rows_to_delete.append(i)

        if not rows_to_delete:
            return

        for i in sorted(rows_to_delete, reverse=True):
            self.roi_polygons.pop(i)

        self.update_roi_table()
        print(f"ROI {len(rows_to_delete)}개 삭제, 남은 ROI: {len(self.roi_polygons)}개")

    # ──────────────────────────────────────────────
    # 프레임 업데이트
    # ──────────────────────────────────────────────
    def update_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        # FPS 계산
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()

        # ── YOLO 추론 ──
        if self.ai_enabled and self.model:
            results = self.model(frame, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names.get(cls_id, str(cls_id))

                        # 경계상자
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 라벨 배경 + 텍스트
                        label = f"{cls_name} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        scaled = pixmap.scaled(
            self.labelCamera.size(),
            aspectRatioMode=1,
            transformMode=1,
        )

        # 오프셋 계산
        label_w = self.labelCamera.width()
        label_h = self.labelCamera.height()
        pix_w = scaled.width()
        pix_h = scaled.height()
        self.pixmap_offset = ((label_w - pix_w) // 2, (label_h - pix_h) // 2)
        self.pixmap_size = (pix_w, pix_h)

        painter = QPainter(scaled)
        painter.setRenderHint(QPainter.Antialiasing)

        for polygon in self.roi_polygons:
            self._draw_polygon(painter, polygon, finished=True)

        if self.roi_mode and self.roi_points:
            self._draw_polygon(painter, self.roi_points, finished=False)

        painter.end()
        self.labelCamera.setPixmap(scaled)

    def _draw_polygon(self, painter, points, finished):
        if finished:
            fill_color = QColor(0, 255, 0, 40)
            edge_color = QColor(0, 255, 0, 200)
            point_color = QColor(0, 255, 0, 255)
        else:
            fill_color = QColor(255, 255, 0, 30)
            edge_color = QColor(255, 255, 0, 200)
            point_color = QColor(255, 255, 0, 255)

        qpoints = [QPoint(int(x), int(y)) for x, y in points]

        if len(qpoints) >= 3:
            painter.setBrush(QBrush(fill_color))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(QPolygon(qpoints))

        pen = QPen(edge_color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for i in range(len(qpoints) - 1):
            painter.drawLine(qpoints[i], qpoints[i + 1])

        if finished and len(qpoints) >= 3:
            painter.drawLine(qpoints[-1], qpoints[0])
        elif not finished and self.current_mouse_pos:
            mouse_pt = QPoint(int(self.current_mouse_pos[0]), int(self.current_mouse_pos[1]))
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(qpoints[-1], mouse_pt)

        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(QBrush(point_color))
        for pt in qpoints:
            painter.drawEllipse(pt, 5, 5)

    def closeEvent(self, event):
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
