import cv2
from convertviedo import *


if __name__ == "__main__":
    image_path_pre = "C:\\Users\\14048\\Desktop\\multimedia\\project\\video_rgb\\labeled_imgs\\Finaltest2_compact/half_hd_foreground_134_"
    frames = []
    # out = cv2.VideoWriter("foreground2.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (910, 486))
    for i in range(603):
        image_path = image_path_pre + "{:03d}".format(i) + ".png"
        frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        for i in range(len(frame)):
            for j in range(len(frame[0])):
                if frame[i][j][3] == 0:
                    frame[i][j] = [255,255,255,255]

        frames.append(frame)

    convert_bgra_2_video(frames, 30)
    #     out.write(frame)
    # out.release()