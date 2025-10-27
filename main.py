import cv2
import numpy as np
import time


def yolo_person_tracking():
    # 初始化摄像头（降低分辨率提高性能）
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # 加载YOLOv4-tiny模型
    net = cv2.dnn.readNet(
        "yolo_files/yolov4-tiny.weights",
        "yolo_files/yolov4-tiny.cfg"
    )

    # 使用OpenCV作为后端，CPU作为目标
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 获取输出层名称
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 加载COCO类别标签
    with open("yolo_files/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # 用于控制检测频率
    detection_interval = 5  # 每5帧检测一次
    frame_count = 0
    person_count = 0
    fps = 0
    last_detection_time = time.time()

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 只在特定间隔进行检测
            if frame_count % detection_interval == 0:
                # 预处理图像
                blob = cv2.dnn.blobFromImage(
                    frame, 0.00392, (320, 320), (0, 0, 0),
                    swapRB=True, crop=False
                )
                net.setInput(blob)
                outs = net.forward(output_layers)

                # 解析检测结果
                person_count = 0
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        # 只处理"person"类别且置信度>50%
                        if confidence > 0.5 and classes[class_id] == 'person':
                            person_count += 1

                            # 计算边界框坐标
                            center_x = int(detection[0] * frame.shape[1])
                            center_y = int(detection[1] * frame.shape[0])
                            w = int(detection[2] * frame.shape[1])
                            h = int(detection[3] * frame.shape[0])

                            # 计算矩形框的左上角坐标
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            # 绘制边界框和置信度
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f"Person: {confidence:.2f}"
                            cv2.putText(frame, label, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 计算FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)

            # 显示统计信息
            cv2.putText(frame, f"Persons: {person_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('YOLOv4-tiny Person Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_person_tracking()