from argparse import ArgumentParser

import cv2
import numpy as np

def lucas_kanade_method(video_path):
    cap = cv2.VideoCapture(video_path)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=10000, qualityLevel=0.001, minDistance=3, blockSize=16)
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(16, 16),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # Create some random colors
    color = np.random.randint(0, 255, (100000, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()



    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    while True:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        # p0 = np.empty(shape=(0, 1, 2), dtype= "float32")
        step = 16 * 4
        for h in range(0, len(old_gray[0]), step):
            for w in range(0, len(old_gray), step):
                block = old_gray[h:h+step][w:w+step]
                features_in_block = cv2.goodFeaturesToTrack(block, mask=None, **feature_params)

                if features_in_block is None:
                    # features_in_block = np.array([[[h+step/4,w+step/4]], [[h+step/4,w+step/4*3]], [[h+step/4*3,w+step/4]], [[h+step/4*3,w+step/4*3]]], dtype = "float32")
                    features_in_block = np.array([[[h+step/4,w+step/4]], [[h+step/4*3,w+step/4]]], dtype = "float32")
                else:
                    for feature in features_in_block:
                        feature[0][0] += w
                        feature[0][1] += h
                p0 = np.append(p0, features_in_block, axis = 0)
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # p1, st, err = cv2.calcOpticalFlowPyrLK(
        #     old_frame, frame, p0, None, **lk_params
        # )
        # Select good points
        # good_new = p1[st == 1]
        # good_old = p0[st == 1]
        good_new = p1
        good_old = p0
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
            # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            # mask = cv2.circle(mask, (int(a), int(b)), 5, color[i].tolist(), -1)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        if k == ord("c"):
            mask = np.zeros_like(old_frame)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # read the video
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = cv2.optflow.calcOpticalFlowSparseToDense(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame

def main():
    # parser = ArgumentParser()
    # parser.add_argument(
    #     "--algorithm",
    #     choices=["farneback", "lucaskanade", "lucaskanade_dense", "rlof"],
    #     required=True,
    #     help="Optical flow algorithm to use",
    # )
    # parser.add_argument(
    #     "--video_path", default="videos/cat.mp4", help="Path to the video",
    # )

    # args = parser.parse_args()
    # # print(args)
    # video_path = args.video_path
    # if args.algorithm == "lucaskanade":
    #     lucas_kanade_method(video_path)
    # elif args.algorithm == "lucaskanade_dense":
    #     method = cv2.optflow.calcOpticalFlowSparseToDense
    #     dense_optical_flow(method, video_path, to_gray=True)
    # elif args.algorithm == "farneback":
    #     method = cv2.calcOpticalFlowFarneback
    #     params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
    #     dense_optical_flow(method, video_path, params, to_gray=True)
    # elif args.algorithm == "rlof":
    #     method = cv2.optflow.calcOpticalFlowDenseRLOF
    #     dense_optical_flow(method, video_path)

    video_path = "../video_view/video1.mp4"

    lucas_kanade_method(video_path)
    # method = cv2.optflow.calcOpticalFlowSparseToDense
    # dense_optical_flow(method, video_path, to_gray=False)


if __name__ == "__main__":
    main()