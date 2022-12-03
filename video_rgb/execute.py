from motion_vector import *
from getPNGPath import *

if __name__ == "__main__":
    # test.py arg1 arg2
    if len(sys.argv) == 2:
        directory = sys.argv[1]
        # sys.argv[2]
    else:
        # directory = "C:/Users/14048/Desktop/multimedia/project/video_rgb/SAL_490_270_437"
        directory = "C:\\Users\\14048\\Desktop\\multimedia\\project\\video_rgb\\SAL_490_270_437"
        splitted1 = directory.split("/")
        splitted2 = []
        for splitted1_seg in splitted1:
            splitted2 += splitted1_seg.split("\\")
        print(splitted2)
    # elif len(sys.argv) == 1:
    #     pass
    # anchor_path = "C:/Users/14048/Desktop/20221130141838.png"
    # target_path = "C:/Users/14048/Desktop/20221130141838.png"
    # main(anchorPath, targetPath, saveOutput=True)
    parent_dict = "/".join(splitted2[:-1]) 
    main_folder = splitted2[-1]

    png_path_prefix = parent_dict + "/" + main_folder + "/" + main_folder

    dict_info = main_folder.split("_")
    width = int(dict_info[1])
    height = int(dict_info[2])
    frame_num = int(dict_info[3])


    # runJavaRGB2PNG(parent_dict, main_folder)

    block_size = 16
    frame_predict_step = 3
    for frame_idx_0 in range(frame_num-frame_predict_step):
        print("calculating motion vector" + str(frame_idx_0))
        frame_idx_1 = frame_idx_0 + frame_predict_step
        frame_idx_0_path = png_path_prefix + "."+  "{:03d}".format(frame_idx_0 + 1) + ".png"
        frame_idx_1_path = png_path_prefix + "."+  "{:03d}".format(frame_idx_1 + 1) + ".png"
        frame_n0 = cv2.imread(frame_idx_0_path)
        frame_n1 = cv2.imread(frame_idx_1_path)
        frame_n0_Y_blockized = preprocess_a_frame(frame_n0, block_size)
        frame_n1_Y_blockized = preprocess_a_frame(frame_n1, block_size)
        motion_vector_matrix, predicted = get_motion_vector_matrix(frame_n0_Y_blockized, frame_n1_Y_blockized, block_size, 8)
        draw_line_on_predicted(predicted, motion_vector_matrix, block_size)
        # save_intermediate_images(frame_n1_Y_blockized, predicted, idx = frame_idx_0)
