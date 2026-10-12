import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from scipy.signal import savgol_filter

# --- 設定區 ---
BALL_MODEL_PATH = r'model_v2\best.pt'
POSE_MODEL_PATH = 'yolo11x-pose.pt'
VIDEO_PATH = r'source_video\test12.mp4'
OUTPUT_PATH = r'test_result.mp4'   
REAL_DISTANCE = 17.5      
CONF = 0.805

# --- SVM 模型設定區 ---
SVM_MODEL_PATH = 'svm_model.pkl'
SVM_SCALER_PATH = 'svm_scaler.pkl'
TARGET_POINTS = 13  # 13個點 = 26個特徵

# ⭐️ 球種對照表
PITCH_NAMES = {0: "Fastball", 1: "Curveball", 2: "Slider"}

# YOLOv8 關鍵點連線
SKELETON_CONNECTIONS = [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14)]

class PitchTracker:
    def __init__(self):
        self.ball_model = YOLO(BALL_MODEL_PATH)
        self.pose_model = YOLO(POSE_MODEL_PATH)
        
        print("正在載入 SVM 球種預測模型...")
        self.svm_model = joblib.load(SVM_MODEL_PATH)
        self.svm_scaler = joblib.load(SVM_SCALER_PATH)
        
        self.trajectory = []  
        self.ball_frames = [] 
        self.strike_zone = None
        self.pitch_result = None      
        self.pitch_type_name = None   
        self.last_detected_idx = -1   
        
    def _draw_debug_skeleton(self, frame, keypoints, color=(0, 255, 255)):
        for s, e in SKELETON_CONNECTIONS:
            x1, y1 = keypoints[s]; x2, y2 = keypoints[e]
            if x1 > 0 and x2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        for j in [5, 6, 11, 12, 13, 14]:
            x, y = keypoints[j]
            if x > 0: cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    def get_dynamic_strike_zone(self, frame):
        """核心幾何邏輯 (Joyboy 專屬參數)"""
        results = self.pose_model(frame, classes=[0], verbose=False)
        if not results or len(results[0].keypoints) == 0: return None
        img_h, img_w = frame.shape[:2]
        candidates = []
        
        for kpts in results[0].keypoints.xy:
            k = kpts.cpu().numpy()
            if len(k) < 15: continue
            sy = [y for y in [k[5][1], k[6][1]] if y > 0]
            hy = [y for y in [k[11][1], k[12][1]] if y > 0]
            ky = [y for y in [k[13][1], k[14][1]] if y > 0]
            ay = [y for y in [k[15][1], k[16][1]] if y > 0]
            if not sy or not hy: continue
            max_sy, max_hy = max(sy), max(hy)
            torso_h = max_hy - min(sy)
            if max_hy < img_h * 0.35 or torso_h < img_h * 0.05: continue
            center_x = (k[5][0] + k[6][0]) / 2 if k[5][0] > 0 else k[0][0]
            lowest_point_y = max(ky + ay + hy) if (ky or ay or hy) else max_hy
            candidates.append({'k': k, 'torso_h': torso_h, 'max_sy': max_sy, 'max_hy': max_hy, 'ky': ky, 'center_x': center_x, 'lowest_point_y': lowest_point_y})
            
        if len(candidates) < 2: return None
        candidates = sorted(candidates, key=lambda x: x['torso_h'], reverse=True)[:4]
        pitcher = max(candidates, key=lambda x: x['lowest_point_y'])
        candidates = [c for c in candidates if c is not pitcher]
        if len(candidates) < 2: return None
        
        sorted_x = sorted(candidates, key=lambda x: x['center_x'])
        if len(sorted_x) >= 3:
            d_left, d_right = abs(sorted_x[1]['center_x'] - sorted_x[0]['center_x']), abs(sorted_x[2]['center_x'] - sorted_x[1]['center_x'])
            if d_left < d_right: ump_catch, batter = sorted_x[0], sorted_x[2]
            else: ump_catch, batter = sorted_x[2], sorted_x[0]
        else:
            img_center_x = img_w / 2
            if abs(sorted_x[0]['center_x'] - img_center_x) > abs(sorted_x[1]['center_x'] - img_center_x): batter, ump_catch = sorted_x[0], sorted_x[1]
            else: batter, ump_catch = sorted_x[1], sorted_x[0]

        if not batter['ky']: return None 
        self._draw_debug_skeleton(frame, batter['k'])
        cv2.putText(frame, "BATTER", (int(batter['center_x'])-40, int(batter['max_sy'])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # --- 📐 專屬計算 ---
        k = batter['k']; torso_h = batter['torso_h']
        top_y = int(batter['max_hy'] - (torso_h * 0.2)) 
        bot_y = int(max(batter['ky']) + (torso_h * 0.25))
        sz_height = bot_y - top_y
        sz_width = int(sz_height * 0.85)
        
        valid_x = [x for x in [k[5][0], k[6][0], k[11][0], k[12][0]] if x > 0]
        b_left, b_right = min(valid_x), max(valid_x)
        GAP = int(sz_width * 0.65) 
        
        if batter['center_x'] < ump_catch['center_x']: sz_cx = b_right + GAP + (sz_width / 2)
        else: sz_cx = b_left - GAP - (sz_width / 2)
            
        return (int(sz_cx - sz_width / 2), top_y, int(sz_cx + sz_width / 2), bot_y)

    def predict_pitch(self):
        """判定好壞球與球種"""
        if not self.strike_zone or len(self.trajectory) < 5: return
        
        # 1. 好壞球
        lx, ly = self.trajectory[-1]
        x1, y1, x2, y2 = self.strike_zone
        self.pitch_result = "STRIKE" if (x1 <= lx <= x2 and y1 <= ly <= y2) else "BALL"
        
        # 2. SVM 預測
        pts = np.array(self.trajectory)
        old_idx = np.linspace(0, 1, len(pts))
        new_idx = np.linspace(0, 1, TARGET_POINTS)
        rx = np.interp(new_idx, old_idx, pts[:, 0])
        ry = np.interp(new_idx, old_idx, pts[:, 1])
        features = np.array([val for pair in zip(rx, ry) for val in pair]).reshape(1, -1)
        
        try:
            scaled_feat = self.svm_scaler.transform(features)
            label = int(self.svm_model.predict(scaled_feat)[0])
            self.pitch_type_name = PITCH_NAMES.get(label, "Unknown")
        except Exception as e:
            print(f"預測錯誤: {e}")

    def process(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        fps, w, h = int(cap.get(5)), int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        results = self.ball_model.track(source=VIDEO_PATH, conf=CONF, iou=0.5, imgsz=1280, stream=True, persist=True)
        strike_zone_locked = False
        
        for idx, r in enumerate(results):
            frame = r.orig_img
            ball_found = False
            
            if not strike_zone_locked:
                new_zone = self.get_dynamic_strike_zone(frame)
                if new_zone: self.strike_zone = new_zone; strike_zone_locked = True
            else:
                self.get_dynamic_strike_zone(frame)
            
            if self.strike_zone:
                x1, y1, x2, y2 = self.strike_zone
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "STRIKE ZONE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if r.boxes.xyxy is not None and len(r.boxes.xyxy) > 0:
                for box in r.boxes.xywh:
                    bx, by, bw, bh = box.cpu().numpy()
                    if bw > 40 or bh > 40: continue
                    self.trajectory.append((int(bx), int(by)))
                    self.ball_frames.append(idx)
                    self.last_detected_idx = idx
                    ball_found = True
                    break 

            if self.last_detected_idx != -1 and not ball_found:
                if (idx - self.last_detected_idx) == 5 and self.pitch_result is None:
                    self.predict_pitch()

            # 繪製軌跡
            if self.trajectory:
                for pt in self.trajectory: cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                if len(self.trajectory) > 5:
                    pts = np.array(self.trajectory)
                    sx, sy = savgol_filter(pts[:, 0], 5, 2), savgol_filter(pts[:, 1], 5, 2)
                    for i in range(1, len(sx)):
                        cv2.line(frame, (int(sx[i-1]), int(sy[i-1])), (int(sx[i]), int(sy[i])), (0, 255, 0), 3)

            # 顯示結果與球種
            if self.pitch_result:
                res_color = (0, 0, 255) if self.pitch_result == "STRIKE" else (0, 255, 0)
                cv2.putText(frame, self.pitch_result, (w//2 - 200, h//2), cv2.FONT_HERSHEY_TRIPLEX, 4.0, res_color, 8)
                if self.pitch_type_name:
                    cv2.putText(frame, f"Pitch: {self.pitch_type_name}", (w//2 - 180, h//2 + 80), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (255, 255, 0), 4)

            # ⭐️ 恢復顯示球速
            if len(self.ball_frames) > 2:
                dur = self.ball_frames[-1] - self.ball_frames[0]
                if dur > 0:
                    speed_kmh = (REAL_DISTANCE / (dur / fps)) * 3.6
                    cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            out.write(frame)
            cv2.imshow("Joyboy AI Scout", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        # 影片結束強制結算
        if self.pitch_result is None and len(self.trajectory) > 0:
            self.predict_pitch()
            
        cap.release(); out.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PitchTracker()
    tracker.process()