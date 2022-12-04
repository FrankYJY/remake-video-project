from motion_vector import *
from getPNGPath import *
from cluster import *

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
        # print(splitted2)
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


    runJavaRGB2PNG(parent_dict+"/", main_folder)

    if_calculate_prediction_and_output = True

    block_size = 16
    search_expand_length = 8
    frame_predict_step = 10
    for frame_idx_0 in range(frame_num-frame_predict_step):
        print("calculating motion vector" + str(frame_idx_0))
        frame_idx_1 = frame_idx_0 + frame_predict_step
        frame_idx_0_path = png_path_prefix + "."+  "{:03d}".format(frame_idx_0 + 1) + ".png"
        frame_idx_1_path = png_path_prefix + "."+  "{:03d}".format(frame_idx_1 + 1) + ".png"
        frame_n0 = cv2.imread(frame_idx_0_path)
        frame_n1 = cv2.imread(frame_idx_1_path)

        # frame_n0 = cv2.resize(frame_n0, (block_size*64, block_size*36), interpolation=cv2.INTER_LINEAR)
        # frame_n1 = cv2.resize(frame_n1, (block_size*64, block_size*36), interpolation=cv2.INTER_LINEAR)
        # preprocess_a_frame_size_inplace(frame_n0, block_size)
        frame_n1 = preprocess_a_frame_size(frame_n1, block_size)
        frame_n0_Y_blockized = preprocess_a_frame_to_Y(frame_n0, block_size)
        frame_n1_Y_blockized = preprocess_a_frame_to_Y(frame_n1, block_size)
        motion_vector_matrix, predicted = get_motion_vector_matrix(frame_n0_Y_blockized, frame_n1_Y_blockized, block_size, "b", search_expand_length, if_calculate_prediction_and_output)
        if if_calculate_prediction_and_output:
            draw_line_on_predicted(predicted, motion_vector_matrix, block_size)
            save_intermediate_images(frame_n1_Y_blockized, predicted, idx = frame_idx_0)
        motion_vector_candidates_set_min_length_to_candidate_idx_0_inplace(motion_vector_matrix)

        # cutting out edges
        motion_vector_matrix_h = len(motion_vector_matrix)
        motion_vector_matrix_w = len(motion_vector_matrix[0])
        motion_vector_matrix = np.array([motion_vector_matrix[x][1:motion_vector_matrix_w-1] for x in range(1,motion_vector_matrix_h-1)])
        cluster_data, motion_vector_xy_to_idx = make_cluster_dataset_of_motion_vectors(motion_vector_matrix)
        # print(cluster_data)

        dydx_tuples = []
        for h in range(len(motion_vector_matrix)):
            for w in range(len(motion_vector_matrix[0])):
                dydx_tuples.append(tuple(motion_vector_matrix[h][w][0]))
        print(Counter(dydx_tuples))


        #motion_vector_matrix = motion_vector_matrix[1:motion_vector_matrix_h-1,1:motion_vector_matrix_w-1,:,:]
        frame_n1_h = len(frame_n1)
        frame_n1_w = len(frame_n1[0])
        frame_n1 = np.array([frame_n1[x][block_size:frame_n1_w-block_size] for x in range(block_size,frame_n1_h-block_size)])
        #frame_n1=frame_n1[block_size:frame_n1_h-block_size,block_size:frame_n1_w-block_size]
        cluster_labels = cluster(cluster_data, 0.5)

        

        labels_descending = get_cluster_labels_descending(cluster_labels)

        frame_n1_RGBA_base = cv2.cvtColor(frame_n1, cv2.COLOR_RGB2RGBA)
        extracted_frame_n1_RGBA_s = []
        for label, label_count in labels_descending:
            frame_n1_RGBA = frame_n1_RGBA_base.copy()
            for w in range(frame_n1_RGBA.shape[1]):
                for h in range(frame_n1_RGBA.shape[0]):
                    h_idx = h // block_size
                    w_idx = w // block_size
                    cur_pixel_belongs_to_block_idx = motion_vector_xy_to_idx[(h_idx, w_idx)]
                    if cluster_labels[cur_pixel_belongs_to_block_idx] == label:
                        frame_n1_RGBA[h][w][3] = 255
                    else:
                        frame_n1_RGBA[h][w] = [0, 255, 0, 255]
            extracted_frame_n1_RGBA_s.append(frame_n1_RGBA)
            cv2.imshow("frame_n1_RGBA" + str(label), frame_n1_RGBA)
            cv2.waitKey(0)
            print("")
