import sys
import time
import math
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

# COCO 17 키포인트 스켈레톤 연결 정의
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # 얼굴
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 상체
    (5, 11), (6, 12), (11, 12),                # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),    # 하체
]

# 키포인트 색상 (BGR) - 부위별 구분
KPT_COLORS = [
    (0, 255, 255),   # 0 nose
    (0, 255, 200),   # 1 left_eye
    (0, 255, 200),   # 2 right_eye
    (0, 200, 255),   # 3 left_ear
    (0, 200, 255),   # 4 right_ear
    (255, 128, 0),   # 5 left_shoulder
    (255, 128, 0),   # 6 right_shoulder
    (255, 200, 0),   # 7 left_elbow
    (255, 200, 0),   # 8 right_elbow
    (255, 255, 0),   # 9 left_wrist
    (255, 255, 0),   # 10 right_wrist
    (0, 255, 0),     # 11 left_hip
    (0, 255, 0),     # 12 right_hip
    (0, 200, 128),   # 13 left_knee
    (0, 200, 128),   # 14 right_knee
    (0, 128, 255),   # 15 left_ankle
    (0, 128, 255),   # 16 right_ankle
]

# 스켈레톤 연결선 색상 (BGR)
LIMB_COLORS = [
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),  # 얼굴
    (255, 128, 0), (255, 200, 0), (255, 255, 0), (255, 200, 0), (255, 255, 0),  # 상체
    (0, 255, 128), (0, 255, 128), (0, 255, 0),   # 몸통
    (0, 200, 128), (0, 128, 255), (0, 200, 128), (0, 128, 255),  # 하체
]


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
                self.model = YOLO('yolo11l-pose.pt')
                self.labelModel.setText("모델: YOLO11l-pose")
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
        self.btnKeypoint.clicked.connect(self.toggle_keypoint_mode)

        # ── AI 분석 상태 ──
        self.ai_enabled = False
        self.keypoint_enabled = False

        # ── 배회 감지용 ──
        self.loiter_tracker = {}   # {track_id: {'enter_time', 'last_seen', 'alerted'}}
        self.LOITER_GONE_SEC = 3.0  # 이 시간 동안 안 보이면 추적 해제

        # ── 침입 감지용 ──
        self.intrusion_logged = set()  # 이미 침입 알림된 track_id

        # ── 쓰러짐 감지용 ──
        self.fall_tracker = {}  # {track_id: {'fall_start', 'alerted'}}

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
            self.loiter_tracker.clear()
            self.intrusion_logged.clear()
            self.fall_tracker.clear()
            print("AI 분석 비활성화")

    def toggle_keypoint_mode(self):
        self.keypoint_enabled = not self.keypoint_enabled
        if self.keypoint_enabled:
            self.labelKeypoint.setText("Keypoint: 활성화")
            self.labelKeypoint.setStyleSheet(
                "QLabel { color: green; font-size: 12px; font-weight: bold; }"
            )
            print("Keypoint 시각화 활성화")
        else:
            self.labelKeypoint.setText("Keypoint: 비활성화")
            self.labelKeypoint.setStyleSheet(
                "QLabel { color: red; font-size: 12px; font-weight: bold; }"
            )
            print("Keypoint 시각화 비활성화")

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
    # 이벤트 감지 시 깜빡임 효과
    # ──────────────────────────────────────────────
    def _flash_widget(self, widget, duration_ms=2000):
        """위젯 배경을 분홍색으로 변경 후 일정 시간 뒤 원래로 복원"""
        original_style = widget.styleSheet()
        flash_style = "QTextEdit { border: 1px solid #dddddd; font-size: 11px; background-color: #f8c8c8; }"
        widget.setStyleSheet(flash_style)
        QTimer.singleShot(duration_ms, lambda: widget.setStyleSheet(original_style))

    # ──────────────────────────────────────────────
    # 배회 이벤트 로그
    # ──────────────────────────────────────────────
    def _log_loitering(self, track_id):
        """배회 감지 시 우측 패널에 로그 기록"""
        kst = QTimeZone(b"Asia/Seoul")
        now = QDateTime.currentDateTime().toTimeZone(kst)
        timestamp = now.toString("HH:mm:ss")
        msg = f"[{timestamp}] ID#{track_id} 배회 감지"
        self.textLoitering.append(msg)
        self._flash_widget(self.textLoitering)
        print(msg)

    def _log_intrusion(self, track_id):
        """침입 감지 시 우측 패널에 로그 기록"""
        kst = QTimeZone(b"Asia/Seoul")
        now = QDateTime.currentDateTime().toTimeZone(kst)
        timestamp = now.toString("HH:mm:ss")
        msg = f"[{timestamp}] ID#{track_id} 침입 감지"
        self.textIntrusion.append(msg)
        self._flash_widget(self.textIntrusion)
        print(msg)

    def _calc_torso_angle(self, kpts):
        """어깨 중심 - 엉덩이 중심 벡터의 수직 대비 각도 계산 (0°=수평, 90°=수직)"""
        conf_thresh = 0.5
        l_sh = kpts[5]  # left_shoulder
        r_sh = kpts[6]  # right_shoulder
        l_hp = kpts[11]  # left_hip
        r_hp = kpts[12]  # right_hip

        # 양쪽 중 하나라도 confidence 부족하면 None
        if float(l_sh[2]) < conf_thresh and float(r_sh[2]) < conf_thresh:
            return None
        if float(l_hp[2]) < conf_thresh and float(r_hp[2]) < conf_thresh:
            return None

        # 유효한 키포인트로 중심점 계산
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

        # atan2(dy, dx) → 수직이면 ~90°, 수평이면 ~0°
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def _log_falldown(self, track_id):
        """쓰러짐 감지 시 우측 패널에 로그 기록"""
        kst = QTimeZone(b"Asia/Seoul")
        now = QDateTime.currentDateTime().toTimeZone(kst)
        timestamp = now.toString("HH:mm:ss")
        msg = f"[{timestamp}] ID#{track_id} 쓰러짐 감지"
        self.textFalldown.append(msg)
        self._flash_widget(self.textFalldown)
        print(msg)

    # ──────────────────────────────────────────────
    # ROI 필터링 (pixmap 좌표 → 원본 프레임 좌표 변환)
    # ──────────────────────────────────────────────
    def _roi_to_frame_coords(self, frame_w, frame_h):
        """ROI 폴리곤들을 pixmap 좌표에서 원본 프레임 좌표로 변환"""
        if not self.roi_polygons or self.pixmap_size[0] == 0:
            return []

        scale_x = frame_w / self.pixmap_size[0]
        scale_y = frame_h / self.pixmap_size[1]

        frame_polygons = []
        for polygon in self.roi_polygons:
            converted = np.array(
                [(int(x * scale_x), int(y * scale_y)) for x, y in polygon],
                dtype=np.int32,
            )
            frame_polygons.append(converted)
        return frame_polygons

    def _is_in_any_roi(self, point, roi_polygons_frame):
        """하단 중심점이 ROI 폴리곤 중 하나라도 내부에 있는지 판정"""
        for poly in roi_polygons_frame:
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return True
        return False

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

        # ── YOLO 추론 + 배회 감지 ──
        if self.ai_enabled and self.model:
            frame_h, frame_w = frame.shape[:2]
            roi_polys = self._roi_to_frame_coords(frame_w, frame_h)
            now = time.time()

            # 추적 모드로 추론 (persist=True: 프레임 간 ID 유지)
            results = self.model.track(frame, persist=True, verbose=False)

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
                    cls_name = self.model.names.get(cls_id, str(cls_id))

                    # 사람(person, cls_id=0)만 처리
                    if cls_id != 0:
                        continue

                    # track ID (추적 실패 시 None)
                    track_id = int(box.id[0]) if box.id is not None else None

                    # 하단 중심점
                    bottom_center = (int((x1 + x2) / 2), y2)

                    # ROI 필터링
                    if roi_polys and not self._is_in_any_roi(bottom_center, roi_polys):
                        continue

                    # 경계상자 색상 (기본 초록, 배회 시 빨강)
                    box_color = (0, 255, 0)

                    # ── 침입 판정 (ROI 진입 즉시) ──
                    if track_id is not None and roi_polys:
                        if track_id not in self.intrusion_logged:
                            self.intrusion_logged.add(track_id)
                            self._log_intrusion(track_id)

                    # ── 배회 판정 (추적 ID 있을 때만) ──
                    if track_id is not None and roi_polys:
                        current_ids_in_roi.add(track_id)

                        if track_id not in self.loiter_tracker:
                            self.loiter_tracker[track_id] = {
                                'enter_time': now,
                                'last_seen': now,
                                'alerted': False,
                            }
                        else:
                            self.loiter_tracker[track_id]['last_seen'] = now

                        dwell = now - self.loiter_tracker[track_id]['enter_time']
                        threshold = self.spinLoiterTime.value()

                        if dwell >= threshold and threshold > 0:
                            box_color = (0, 0, 255)  # 빨강

                            if not self.loiter_tracker[track_id]['alerted']:
                                self.loiter_tracker[track_id]['alerted'] = True
                                self._log_loitering(track_id)

                    # 경계상자 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(frame, bottom_center, 4, (0, 0, 255), -1)

                    # 라벨
                    id_str = f" ID:{track_id}" if track_id is not None else ""
                    label = f"{cls_name}{id_str} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    # ── 키포인트 + 스켈레톤 시각화 ──
                    if self.keypoint_enabled and keypoints is not None and i < len(keypoints):
                        kpts = keypoints[i].data[0]  # [17, 3] (x, y, conf)
                        self._draw_keypoints(frame, kpts)

                    # ── 쓰러짐 판정 (키포인트 기반) ──
                    if track_id is not None and keypoints is not None and i < len(keypoints):
                        kpts = keypoints[i].data[0]
                        angle = self._calc_torso_angle(kpts)
                        angle_thresh = self.spinFallAngle.value()
                        time_thresh = self.spinFallTime.value()

                        if angle is not None and angle < angle_thresh:
                            # 쓰러짐 후보 상태
                            if track_id not in self.fall_tracker:
                                self.fall_tracker[track_id] = {
                                    'fall_start': now,
                                    'alerted': False,
                                }
                            fall_dur = now - self.fall_tracker[track_id]['fall_start']
                            if fall_dur >= time_thresh and not self.fall_tracker[track_id]['alerted']:
                                self.fall_tracker[track_id]['alerted'] = True
                                self._log_falldown(track_id)
                        else:
                            # 정상 자세로 복귀 → 초기화
                            if track_id in self.fall_tracker:
                                del self.fall_tracker[track_id]

            # ── 사라진 ID 정리 ──
            gone_ids = []
            for tid, info in self.loiter_tracker.items():
                if tid not in current_ids_in_roi:
                    if now - info['last_seen'] > self.LOITER_GONE_SEC:
                        gone_ids.append(tid)
            for tid in gone_ids:
                del self.loiter_tracker[tid]
                self.fall_tracker.pop(tid, None)

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

    def _draw_keypoints(self, frame, kpts):
        """키포인트 점 + 스켈레톤 연결선 시각화"""
        kpt_conf_thresh = 0.5

        # 스켈레톤 연결선 그리기
        for idx, (a, b) in enumerate(SKELETON):
            xa, ya, ca = int(kpts[a][0]), int(kpts[a][1]), float(kpts[a][2])
            xb, yb, cb = int(kpts[b][0]), int(kpts[b][1]), float(kpts[b][2])
            if ca > kpt_conf_thresh and cb > kpt_conf_thresh:
                color = LIMB_COLORS[idx] if idx < len(LIMB_COLORS) else (200, 200, 200)
                cv2.line(frame, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)

        # 키포인트 점 그리기
        for j in range(len(kpts)):
            x, y, c = int(kpts[j][0]), int(kpts[j][1]), float(kpts[j][2])
            if c > kpt_conf_thresh:
                color = KPT_COLORS[j] if j < len(KPT_COLORS) else (255, 255, 255)
                cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)

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
