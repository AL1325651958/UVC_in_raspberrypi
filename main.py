import cv2
import numpy as np
import time
import threading


# 摄像头读取线程（防止阻塞）
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


# 主检测逻辑
def yolo_person_tracking_optimized():
    print("加载 YOLOv4-tiny 模型中...")
    net = cv2.dnn.readNet(
        "yolo_files/yolov4-tiny.weights",
        "yolo_files/yolov4-tiny.cfg"
    )

    # 后端使用 CPU（OpenCV DNN）
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 获取输出层名
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # 载入 COCO 类别
    with open("yolo_files/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # 启动摄像头线程
    cam = CameraThread(width=640, height=480)
    time.sleep(1.0)

    detection_interval = 8  # 每8帧检测一次
    frame_count = 0
    fps_smooth = 0
    person_boxes = []  # 存储框
    conf_threshold = 0.45
    nms_threshold = 0.4  # IOU重叠阈值

    print("检测启动，按 Q 退出...")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            start_time = time.time()
            frame_count += 1
            height, width = frame.shape[:2]

            # 每隔 detection_interval 帧进行一次检测
            if frame_count % detection_interval == 0:
                # 缩放输入到 160x160 提高推理速度
                blob = cv2.dnn.blobFromImage(
                    frame, 1/255.0, (160, 160), (0, 0, 0),
                    swapRB=True, crop=False
                )
                net.setInput(blob)
                outs = net.forward(output_layers)

                boxes = []
                confidences = []

                # 解析 YOLO 输出
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > conf_threshold and classes[class_id] == 'person':
                            # 反算原图坐标
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))

                # 非极大值抑制 (NMS)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                person_boxes.clear()
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        conf = confidences[i]
                        person_boxes.append((x, y, w, h, conf))

            # 绘制检测框
            for (x, y, w, h, conf) in person_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 230, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 0), 1)

            # FPS 计算（平滑）
            fps = 1.0 / (time.time() - start_time)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

            # 显示统计信息
            cv2.putText(frame, f"Persons: {len(person_boxes)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 缩小显示窗口减负
            disp = cv2.resize(frame, (480, 360))
            cv2.imshow("YOLOv4-tiny Optimized (160x160 + NMS)", disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_person_tracking_optimized()
