import numpy as np
import cv2

def convert_video_2_bgra(video_path):

    cap = cv2.VideoCapture(video_path)
    arr = []
    if not cap.isOpened():
        print("Error: video can not load")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        arr.append(bgra)
        #arr.append(frame)
        if(cv2.waitKey(1)==ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

    return (arr,fps)

def convert_bgra_2_video(arr, fps):

    height, width = len(arr[0]), len(arr[0][0])
    out = cv2.VideoWriter("new_video1.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    for i in range(len(arr)):
        out.write(cv2.cvtColor(arr[i], cv2.COLOR_BGRA2BGR))
    out.release()


if __name__ == "__main__":
    video_path = "/Users/zihao/Documents/GitHub/remake-video-project/video_view/video2.mp4"
    arr,fps = convert_video_2_bgra(video_path)
    print(fps)
    convert_bgra_2_video(arr, fps)
