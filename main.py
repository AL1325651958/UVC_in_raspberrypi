import cv2
import numpy as np
import time
import threading

# ----------------- 摄像头后台读取线程 -----------------
class CameraThread:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(3, width)
        self.cap.set(4, height)
        self.ret = False
        self.frame = None
        self.stopped = False
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# ----------------- 主检测函数 -----------------
def yolo_person_tracking_optimized():
    print("初始化 YOLOv4-tiny 模型中...")
    net = cv2.dnn.readNet(
        "yolo_files/yolov4-tiny.weights",
        "yolo_files/yolov4-tiny.cfg"
    )
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 读取输出层
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # 加载类别名称
    with open("yolo_files/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # 初始化摄像头线程
    cam = CameraThread(width=640, height=480)
    time.sleep(1.0)  # 等待摄像头稳定

    detection_interval = 8  # 每8帧检测一次
    frame_count = 0
    fps_smooth = 0
    person_boxes = []  # 存储上一次检测结果
    last_detection_time = 0

    print("检测启动，按 Q 退出")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            start_time = time.time()
            frame_count += 1

            # 每隔 detection_interval 帧进行检测
            if frame_count % detection_interval == 0:
                blob = cv2.dnn.blobFromImage(
                    frame, 0.00392, (256, 256), (0, 0, 0),
                    swapRB=True, crop=False
                )
                net.setInput(blob)
                outs = net.forward(output_layers)

                person_boxes.clear()
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.45 and classes[class_id] == 'person':
                            center_x = int(detection[0] * frame.shape[1])
                            center_y = int(detection[1] * frame.shape[0])
                            w = int(detection[2] * frame.shape[1])
                            h = int(detection[3] * frame.shape[0])
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            person_boxes.append((x, y, w, h, confidence))

                last_detection_time = time.time()

            # 绘制检测框
            for (x, y, w, h, conf) in person_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

            # 计算平滑FPS
            fps = 1.0 / (time.time() - start_time)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

            # 显示统计信息
            cv2.putText(frame, f"Persons: {len(person_boxes)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 缩小显示窗口尺寸以减轻渲染负担
            disp_frame = cv2.resize(frame, (480, 360))
            cv2.imshow("YOLOv4-tiny Person Tracking (Optimized)", disp_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_person_tracking_optimized()
