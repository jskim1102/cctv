import sys
import time
import math
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidgetItem, QCheckBox,
    QWidget, QHBoxLayout, QHeaderView, QAbstractItemView, QPushButton,
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

MODEL_PATH = 'yolo11l-pose.pt'

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

KPT_COLORS = [
    (0, 255, 255), (0, 255, 200), (0, 255, 200),
    (0, 200, 255), (0, 200, 255),
    (255, 128, 0), (255, 128, 0),
    (255, 200, 0), (255, 200, 0),
    (255, 255, 0), (255, 255, 0),
    (0, 255, 0), (0, 255, 0),
    (0, 200, 128), (0, 200, 128),
    (0, 128, 255), (0, 128, 255),
]

LIMB_COLORS = [
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (255, 128, 0), (255, 200, 0), (255, 255, 0), (255, 200, 0), (255, 255, 0),
    (0, 255, 128), (0, 255, 128), (0, 255, 0),
    (0, 200, 128), (0, 128, 255), (0, 200, 128), (0, 128, 255),
]

MAX_CAMS = 4
LOITER_GONE_SEC = 3.0
CAM_STYLE_SELECTED = "QLabel { background-color: #2d2d2d; color: #aaaaaa; font-size: 20px; font-weight: bold; border: 3px solid #00ff00; }"
CAM_STYLE_NORMAL = "QLabel { background-color: #2d2d2d; color: #aaaaaa; font-size: 20px; font-weight: bold; border: 1px solid #555555; }"


class MainWindow(base_class, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._init_roi_table()

        # FPS
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # 카메라 (4슬롯)
        self.cam_labels = [self.labelCam1, self.labelCam2, self.labelCam3, self.labelCam4]
        self.caps = [None] * MAX_CAMS
        self.cam_hw_indices = [-1] * MAX_CAMS
        self.selected_cam = 0
        self.pixmap_offsets = [(0, 0)] * MAX_CAMS
        self.pixmap_sizes = [(0, 0)] * MAX_CAMS
        self.failed_hw_indices = set()  # 연결 실패한 hw 인덱스 (재시도 방지)

        # YOLO 모델 (카메라별 독립 인스턴스)
        self.models = [None] * MAX_CAMS
        self.model_loaded = False
        if HAS_YOLO:
            try:
                test_model = YOLO(MODEL_PATH)
                self.labelModel.setText("모델: YOLO11l-pose")
                self.labelModel.setStyleSheet(
                    "QLabel { color: green; font-size: 12px; font-weight: bold; }"
                )
                self.model_loaded = True
                del test_model
                print("YOLO 모델 로드 확인 완료")
            except Exception as e:
                print(f"모델 로드 실패: {e}")
                self.labelModel.setText("모델: 로드실패")
        else:
            self.labelModel.setText("모델: ultralytics 미설치")

        self._scan_cameras()

        # GPU
        if HAS_TORCH and torch.cuda.is_available():
            self.labelGPU.setText(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.labelGPU.setText("GPU: 사용불가")

        # ROI (카메라별)
        self.roi_mode = False
        self.roi_points = []
        self.roi_polygons = [[] for _ in range(MAX_CAMS)]
        self.current_mouse_pos = None

        # 버튼
        self.btnROI.clicked.connect(self.toggle_roi_mode)
        self.btnDeleteROI.clicked.connect(self.delete_selected_roi)
        self.btnAI.clicked.connect(self.toggle_ai_mode)
        self.btnKeypoint.clicked.connect(self.toggle_keypoint_mode)
        self.btnConnect.clicked.connect(self._manual_scan)

        # AI 상태
        self.ai_enabled = False
        self.keypoint_enabled = False

        # 이벤트 트래커 (카메라별 독립)
        self.loiter_trackers = [{} for _ in range(MAX_CAMS)]
        self.intrusion_logged = [set() for _ in range(MAX_CAMS)]
        self.fall_trackers = [{} for _ in range(MAX_CAMS)]

        # 마우스
        for idx, label in enumerate(self.cam_labels):
            label.setMouseTracking(True)
            label.mousePressEvent = lambda e, i=idx: self._cam_mouse_press(e, i)
            label.mouseMoveEvent = lambda e, i=idx: self._cam_mouse_move(e, i)

        self._update_cam_borders()

        # ── 확대/축소 ──
        self.zoomed_cam = -1  # -1이면 4분할, 0~3이면 해당 카메라 확대
        self.zoom_buttons = []
        btn_style = ("QPushButton { font-size: 11px; font-weight: bold; padding: 4px 10px; "
                      "background-color: rgba(0,0,0,180); color: white; border: 1px solid #888; "
                      "border-radius: 3px; } QPushButton:hover { background-color: rgba(50,50,50,200); }")
        for idx, label in enumerate(self.cam_labels):
            btn = QPushButton("확대", label)
            btn.setStyleSheet(btn_style)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, i=idx: self._zoom_in(i))
            btn.raise_()
            self.zoom_buttons.append(btn)

        self.shrink_btn = QPushButton("축소", self.cam_labels[0])
        self.shrink_btn.setStyleSheet(btn_style)
        self.shrink_btn.setCursor(Qt.PointingHandCursor)
        self.shrink_btn.clicked.connect(self._zoom_out)
        self.shrink_btn.hide()

        # 타이머
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)
        self.update_status()

        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self._scan_cameras)
        self.scan_timer.start(5000)

    def _init_roi_table(self):
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

    # ── 카메라 스캔 ──
    def _scan_cameras(self):
        used_hw = {self.cam_hw_indices[s] for s in range(MAX_CAMS) if self.caps[s] is not None}

        for slot in range(MAX_CAMS):
            if self.caps[slot] is not None and not self.caps[slot].isOpened():
                print(f"cam{slot + 1} 연결 끊김")
                self.caps[slot] = None
                self.cam_hw_indices[slot] = -1
                self._clear_cam_trackers(slot)
                if self.models[slot] is not None:
                    del self.models[slot]
                    self.models[slot] = None

        empty_slots = [s for s in range(MAX_CAMS) if self.caps[s] is None]
        if not empty_slots:
            self._update_camera_status()
            return

        for hw_idx in range(5):
            if hw_idx in used_hw or hw_idx in self.failed_hw_indices or not empty_slots:
                continue
            cap = cv2.VideoCapture(hw_idx, cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    slot = empty_slots.pop(0)
                    self.caps[slot] = cap
                    self.cam_hw_indices[slot] = hw_idx
                    used_hw.add(hw_idx)
                    if self.model_loaded and self.models[slot] is None:
                        self.models[slot] = YOLO(MODEL_PATH)
                        print(f"cam{slot + 1} 모델 인스턴스 로드")
                    print(f"카메라 hw:{hw_idx} -> cam{slot + 1}")
                else:
                    cap.release()
                    self.failed_hw_indices.add(hw_idx)
            else:
                cap.release()
                self.failed_hw_indices.add(hw_idx)

        self._update_camera_status()

    def _update_camera_status(self):
        connected = sum(1 for c in self.caps if c is not None)
        self.labelCameraStatus.setText(f"카메라: {connected}/{MAX_CAMS} 연결됨")
        color = "green" if connected > 0 else "red"
        self.labelCameraStatus.setStyleSheet(
            f"QLabel {{ color: {color}; font-size: 12px; font-weight: bold; }}"
        )

    def _manual_scan(self):
        """카메라 연결 버튼 → 실패 목록 초기화 후 재스캔"""
        self.failed_hw_indices.clear()
        self._scan_cameras()

    def _clear_cam_trackers(self, slot):
        self.loiter_trackers[slot].clear()
        self.intrusion_logged[slot].clear()
        self.fall_trackers[slot].clear()

    # ── 카메라 선택 ──
    def _select_camera(self, cam_idx):
        if cam_idx == self.selected_cam:
            return
        self.selected_cam = cam_idx
        self.update_roi_table()
        self._update_cam_borders()
        print(f"cam{cam_idx + 1} 선택됨")

    def _update_cam_borders(self):
        for idx, label in enumerate(self.cam_labels):
            label.setStyleSheet(CAM_STYLE_SELECTED if idx == self.selected_cam else CAM_STYLE_NORMAL)

    def _zoom_in(self, slot):
        """4분할 → 1분할 확대"""
        self.zoomed_cam = slot
        self._select_camera(slot)

        grid = self.cameraGridLayout
        # 모든 라벨을 그리드에서 제거
        for i, label in enumerate(self.cam_labels):
            grid.removeWidget(label)
            if i != slot:
                label.hide()

        # 선택된 카메라만 전체 영역에 배치
        grid.addWidget(self.cam_labels[slot], 0, 0, 2, 2)
        self.cam_labels[slot].show()

        # 확대 버튼 숨기기, 축소 버튼 표시
        for btn in self.zoom_buttons:
            btn.hide()
        self.shrink_btn.setParent(self.cam_labels[slot])
        self.shrink_btn.show()
        self.shrink_btn.raise_()

        self._reposition_zoom_buttons()

    def _zoom_out(self):
        """1분할 → 4분할 복원"""
        grid = self.cameraGridLayout
        # 확대된 카메라 제거
        if self.zoomed_cam >= 0:
            grid.removeWidget(self.cam_labels[self.zoomed_cam])

        # 원래 위치에 모든 라벨 복원
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, (r, c) in enumerate(positions):
            grid.addWidget(self.cam_labels[i], r, c)
            self.cam_labels[i].show()

        self.zoomed_cam = -1

        # 축소 버튼 숨기기, 확대 버튼 표시
        self.shrink_btn.hide()
        for btn in self.zoom_buttons:
            btn.show()
            btn.raise_()

        self._reposition_zoom_buttons()

    def _reposition_zoom_buttons(self):
        """확대/축소 버튼 위치를 라벨 크기에 맞게 재배치"""
        margin = 8
        for idx, btn in enumerate(self.zoom_buttons):
            label = self.cam_labels[idx]
            btn.adjustSize()
            bw, bh = btn.width(), btn.height()
            btn.move(label.width() - bw - margin, label.height() - bh - margin)

        if self.zoomed_cam >= 0:
            label = self.cam_labels[self.zoomed_cam]
            self.shrink_btn.adjustSize()
            bw, bh = self.shrink_btn.width(), self.shrink_btn.height()
            self.shrink_btn.move(label.width() - bw - margin, margin)

    # ── 상태바 ──
    def update_status(self):
        self.labelFPS.setText(f"FPS: {self.fps:.1f}")
        kst = QTimeZone(b"Asia/Seoul")
        now = QDateTime.currentDateTime().toTimeZone(kst)
        self.labelDateTime.setText(now.toString("yyyy-MM-dd HH:mm:ss"))

    def _label_to_pixmap(self, x, y, slot=None):
        if slot is None:
            slot = self.selected_cam
        px = x - self.pixmap_offsets[slot][0]
        py = y - self.pixmap_offsets[slot][1]
        pw, ph = self.pixmap_sizes[slot]
        if pw > 0 and ph > 0:
            px = max(0, min(px, pw - 1))
            py = max(0, min(py, ph - 1))
        return (px, py)

    # ── 토글 ──
    def toggle_roi_mode(self):
        if not self.roi_mode:
            self.roi_mode = True
            self.roi_points = []
            self.btnROI.setText("ROI 완료")
            self.cam_labels[self.selected_cam].setCursor(Qt.CrossCursor)
        else:
            self.roi_mode = False
            if len(self.roi_points) >= 3:
                self.roi_polygons[self.selected_cam].append(list(self.roi_points))
                self.update_roi_table()
            self.roi_points = []
            self.btnROI.setText("ROI 설정")
            self.cam_labels[self.selected_cam].setCursor(Qt.ArrowCursor)

    def toggle_ai_mode(self):
        if not self.model_loaded:
            print("YOLO 모델이 로드되지 않았습니다.")
            return
        self.ai_enabled = not self.ai_enabled
        if self.ai_enabled:
            self.labelAI.setText("AI 분석: 활성화")
            self.labelAI.setStyleSheet("QLabel { color: green; font-size: 12px; font-weight: bold; }")
        else:
            self.labelAI.setText("AI 분석: 비활성화")
            self.labelAI.setStyleSheet("QLabel { color: red; font-size: 12px; font-weight: bold; }")
            for s in range(MAX_CAMS):
                self._clear_cam_trackers(s)

    def toggle_keypoint_mode(self):
        self.keypoint_enabled = not self.keypoint_enabled
        if self.keypoint_enabled:
            self.labelKeypoint.setText("Keypoint: 활성화")
            self.labelKeypoint.setStyleSheet("QLabel { color: green; font-size: 12px; font-weight: bold; }")
        else:
            self.labelKeypoint.setText("Keypoint: 비활성화")
            self.labelKeypoint.setStyleSheet("QLabel { color: red; font-size: 12px; font-weight: bold; }")

    # ── 마우스 ──
    def _cam_mouse_press(self, event, cam_idx):
        if not self.roi_mode:
            if event.button() == Qt.LeftButton:
                self._select_camera(cam_idx)
            return
        if cam_idx != self.selected_cam:
            return
        if event.button() == Qt.LeftButton:
            px, py = self._label_to_pixmap(event.pos().x(), event.pos().y(), cam_idx)
            self.roi_points.append((px, py))
        elif event.button() == Qt.RightButton:
            if len(self.roi_points) >= 3:
                self.roi_polygons[self.selected_cam].append(list(self.roi_points))
                self.update_roi_table()
            self.roi_points = []
            self.roi_mode = False
            self.btnROI.setText("ROI 설정")
            self.cam_labels[self.selected_cam].setCursor(Qt.ArrowCursor)

    def _cam_mouse_move(self, event, cam_idx):
        if self.roi_mode and cam_idx == self.selected_cam:
            px, py = self._label_to_pixmap(event.pos().x(), event.pos().y(), cam_idx)
            self.current_mouse_pos = (px, py)

    # ── ROI 테이블 ──
    def update_roi_table(self):
        polys = self.roi_polygons[self.selected_cam]
        self.tableROI.setRowCount(len(polys))
        for i, polygon in enumerate(polys):
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
        rows = []
        for i in range(self.tableROI.rowCount()):
            w = self.tableROI.cellWidget(i, 0)
            if w:
                cb = w.findChild(QCheckBox)
                if cb and cb.isChecked():
                    rows.append(i)
        if not rows:
            return
        for i in sorted(rows, reverse=True):
            self.roi_polygons[self.selected_cam].pop(i)
        self.update_roi_table()

    # ── 깜빡임 ──
    def _flash_widget(self, widget, duration_ms=2000):
        original_style = widget.styleSheet()
        widget.setStyleSheet("QTextEdit { border: 1px solid #dddddd; font-size: 11px; background-color: #f8c8c8; }")
        QTimer.singleShot(duration_ms, lambda: widget.setStyleSheet(original_style))

    # ── 이벤트 로그 ──
    def _get_timestamp(self):
        kst = QTimeZone(b"Asia/Seoul")
        return QDateTime.currentDateTime().toTimeZone(kst).toString("HH:mm:ss")

    def _log_loitering(self, track_id, slot):
        msg = f"[{self._get_timestamp()}] ID#{track_id} 배회 감지 (cam{slot + 1})"
        self.textLoitering.append(msg)
        self._flash_widget(self.textLoitering)
        print(msg)

    def _log_intrusion(self, track_id, slot):
        msg = f"[{self._get_timestamp()}] ID#{track_id} 침입 감지 (cam{slot + 1})"
        self.textIntrusion.append(msg)
        self._flash_widget(self.textIntrusion)
        print(msg)

    def _log_falldown(self, track_id, slot):
        msg = f"[{self._get_timestamp()}] ID#{track_id} 쓰러짐 감지 (cam{slot + 1})"
        self.textFalldown.append(msg)
        self._flash_widget(self.textFalldown)
        print(msg)

    # ── 몸통 각도 ──
    def _calc_torso_angle(self, kpts):
        conf_thresh = 0.5
        l_sh, r_sh = kpts[5], kpts[6]
        l_hp, r_hp = kpts[11], kpts[12]
        if float(l_sh[2]) < conf_thresh and float(r_sh[2]) < conf_thresh:
            return None
        if float(l_hp[2]) < conf_thresh and float(r_hp[2]) < conf_thresh:
            return None
        sh_pts = []
        if float(l_sh[2]) >= conf_thresh:
            sh_pts.append((float(l_sh[0]), float(l_sh[1])))
        if float(r_sh[2]) >= conf_thresh:
            sh_pts.append((float(r_sh[0]), float(r_sh[1])))
        sh_cx = sum(p[0] for p in sh_pts) / len(sh_pts)
        sh_cy = sum(p[1] for p in sh_pts) / len(sh_pts)
        hp_pts = []
        if float(l_hp[2]) >= conf_thresh:
            hp_pts.append((float(l_hp[0]), float(l_hp[1])))
        if float(r_hp[2]) >= conf_thresh:
            hp_pts.append((float(r_hp[0]), float(r_hp[1])))
        hp_cx = sum(p[0] for p in hp_pts) / len(hp_pts)
        hp_cy = sum(p[1] for p in hp_pts) / len(hp_pts)
        dx = abs(hp_cx - sh_cx)
        dy = abs(hp_cy - sh_cy)
        return math.degrees(math.atan2(dy, dx))

    # ── ROI 필터링 ──
    def _roi_to_frame_coords(self, frame_w, frame_h, slot):
        if not self.roi_polygons[slot] or self.pixmap_sizes[slot][0] == 0:
            return []
        scale_x = frame_w / self.pixmap_sizes[slot][0]
        scale_y = frame_h / self.pixmap_sizes[slot][1]
        frame_polygons = []
        for polygon in self.roi_polygons[slot]:
            converted = np.array(
                [(int(x * scale_x), int(y * scale_y)) for x, y in polygon],
                dtype=np.int32,
            )
            frame_polygons.append(converted)
        return frame_polygons

    def _is_in_any_roi(self, point, roi_polygons_frame):
        for poly in roi_polygons_frame:
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return True
        return False

    # ── 프레임 업데이트 ──
    def update_frame(self):
        now = time.time()
        for slot in range(MAX_CAMS):
            cap = self.caps[slot]
            label = self.cam_labels[slot]
            if cap is None:
                continue
            ret, frame = cap.read()
            if not ret:
                continue

            is_selected = (slot == self.selected_cam)

            if is_selected:
                self.frame_count += 1
                elapsed = now - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = now

            # YOLO 추론 (모든 카메라)
            if self.ai_enabled and self.models[slot] is not None:
                self._process_detection(frame, slot, now)

            # QPixmap 변환 (보이는 카메라만)
            if not label.isVisible():
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            scaled = pixmap.scaled(label.size(), aspectRatioMode=1, transformMode=1)

            lw, lh = label.width(), label.height()
            pw, ph = scaled.width(), scaled.height()
            self.pixmap_offsets[slot] = ((lw - pw) // 2, (lh - ph) // 2)
            self.pixmap_sizes[slot] = (pw, ph)

            # ROI 오버레이
            has_roi = bool(self.roi_polygons[slot])
            has_drawing = is_selected and self.roi_mode and self.roi_points
            if has_roi or has_drawing:
                painter = QPainter(scaled)
                painter.setRenderHint(QPainter.Antialiasing)
                for polygon in self.roi_polygons[slot]:
                    self._draw_polygon(painter, polygon, finished=True)
                if has_drawing:
                    self._draw_polygon(painter, self.roi_points, finished=False)
                painter.end()

            label.setPixmap(scaled)

        # 버튼 위치 업데이트
        self._reposition_zoom_buttons()

    # ── 카메라별 감지 처리 ──
    def _process_detection(self, frame, slot, now):
        model = self.models[slot]
        frame_h, frame_w = frame.shape[:2]
        roi_polys = self._roi_to_frame_coords(frame_w, frame_h, slot)

        loiter = self.loiter_trackers[slot]
        intrusion = self.intrusion_logged[slot]
        fall = self.fall_trackers[slot]

        results = model.track(frame, persist=True, verbose=False)
        current_ids_in_roi = set()

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))
                if cls_id != 0:
                    continue

                track_id = int(box.id[0]) if box.id is not None else None
                bottom_center = (int((x1 + x2) / 2), y2)

                if roi_polys and not self._is_in_any_roi(bottom_center, roi_polys):
                    continue

                box_color = (0, 255, 0)

                # 침입
                if track_id is not None and roi_polys:
                    if track_id not in intrusion:
                        intrusion.add(track_id)
                        self._log_intrusion(track_id, slot)

                # 배회
                if track_id is not None and roi_polys:
                    current_ids_in_roi.add(track_id)
                    if track_id not in loiter:
                        loiter[track_id] = {'enter_time': now, 'last_seen': now, 'alerted': False}
                    else:
                        loiter[track_id]['last_seen'] = now
                    dwell = now - loiter[track_id]['enter_time']
                    threshold = self.spinLoiterTime.value()
                    if dwell >= threshold and threshold > 0:
                        box_color = (0, 0, 255)
                        if not loiter[track_id]['alerted']:
                            loiter[track_id]['alerted'] = True
                            self._log_loitering(track_id, slot)

                # 경계상자
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, bottom_center, 4, (0, 0, 255), -1)
                id_str = f" ID:{track_id}" if track_id is not None else ""
                label_text = f"{cls_name}{id_str} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
                cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # 키포인트
                if self.keypoint_enabled and keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i].data[0]
                    self._draw_keypoints(frame, kpts)

                # 쓰러짐
                if track_id is not None and keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i].data[0]
                    angle = self._calc_torso_angle(kpts)
                    angle_thresh = self.spinFallAngle.value()
                    time_thresh = self.spinFallTime.value()
                    if angle is not None and angle < angle_thresh:
                        if track_id not in fall:
                            fall[track_id] = {'fall_start': now, 'alerted': False}
                        fall_dur = now - fall[track_id]['fall_start']
                        if fall_dur >= time_thresh and not fall[track_id]['alerted']:
                            fall[track_id]['alerted'] = True
                            self._log_falldown(track_id, slot)
                    else:
                        fall.pop(track_id, None)

        # 사라진 ID 정리
        gone = [t for t, info in loiter.items()
                if t not in current_ids_in_roi and now - info['last_seen'] > LOITER_GONE_SEC]
        for tid in gone:
            del loiter[tid]
            fall.pop(tid, None)

    # ── 키포인트 시각화 ──
    def _draw_keypoints(self, frame, kpts):
        c_th = 0.5
        for idx, (a, b) in enumerate(SKELETON):
            xa, ya, ca = int(kpts[a][0]), int(kpts[a][1]), float(kpts[a][2])
            xb, yb, cb = int(kpts[b][0]), int(kpts[b][1]), float(kpts[b][2])
            if ca > c_th and cb > c_th:
                color = LIMB_COLORS[idx] if idx < len(LIMB_COLORS) else (200, 200, 200)
                cv2.line(frame, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)
        for j in range(len(kpts)):
            x, y, c = int(kpts[j][0]), int(kpts[j][1]), float(kpts[j][2])
            if c > c_th:
                color = KPT_COLORS[j] if j < len(KPT_COLORS) else (255, 255, 255)
                cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)

    # ── ROI 폴리곤 ──
    def _draw_polygon(self, painter, points, finished):
        if finished:
            fill_c, edge_c, pt_c = QColor(0,255,0,40), QColor(0,255,0,200), QColor(0,255,0,255)
        else:
            fill_c, edge_c, pt_c = QColor(255,255,0,30), QColor(255,255,0,200), QColor(255,255,0,255)
        qpts = [QPoint(int(x), int(y)) for x, y in points]
        if len(qpts) >= 3:
            painter.setBrush(QBrush(fill_c))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(QPolygon(qpts))
        pen = QPen(edge_c, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for i in range(len(qpts) - 1):
            painter.drawLine(qpts[i], qpts[i + 1])
        if finished and len(qpts) >= 3:
            painter.drawLine(qpts[-1], qpts[0])
        elif not finished and self.current_mouse_pos:
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(qpts[-1], QPoint(int(self.current_mouse_pos[0]), int(self.current_mouse_pos[1])))
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(QBrush(pt_c))
        for pt in qpts:
            painter.drawEllipse(pt, 5, 5)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._reposition_zoom_buttons)

    def closeEvent(self, event):
        self.timer.stop()
        self.status_timer.stop()
        self.scan_timer.stop()
        for cap in self.caps:
            if cap and cap.isOpened():
                cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
