import cv2
def capture_video_stream():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    
    if not cap.isOpened():
        print("error")
        return

    try:
        while True:
            ret,frame = cap.read()
            
            if not ret:
                print("no ret")
                break
            
            cv2.imshow("UVC",frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    capture_video_stream()