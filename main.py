import cv2
import numpy as np
import time


def lightweight_person_motion_tracking():
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    # 背景建模器（MOG2比帧差更稳）
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

    fps = 0
    person_count = 0
    last_time = time.time()

    print("轻量级人形检测已启动（按 Q 退出）")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 缩小尺寸以提速
        frame_small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        mask = bg_subtractor.apply(gray)

        # 形态学处理消除噪声
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=4)

        # 查找移动目标轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        person_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:  # 忽略小噪点
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            person_count += 1
            cv2.rectangle(frame_small, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 计算FPS
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now

        # 显示结果
        cv2.putText(frame_small, f"Persons: {person_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_small, f"FPS: {fps:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Motion Detection (CM0 Optimized)", frame_small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    lightweight_person_motion_tracking()
