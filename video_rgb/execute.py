from motion_vector import *
from getPNGPath import *
from cluster import *
from convertviedo import *
from panorama import *
import multiprocessing
import pickle
from functools import partial

def main_loop(frame_idx_0, frame_predict_step, input_type, png_path_prefix, frames, resizeT_cutoutF, motion_vector_storage, over_this_threshold_count_set_as_search_start_bkg_threshold, search_method, max_best_candidates_per_level, motion_difference_threshold_search_directions,motion_difference_tolerate_thresholds, main_folder, block_size, search_expand_length, if_calculate_prediction_and_output, all_resolution_frames, labeled_generation_mode):
    print("start", frame_idx_0)
    jump_generated_hd_imgs = False
    if jump_generated_hd_imgs and os.path.exists("./labeled_imgs/"+ main_folder + "/hd_background_0_"+"{:03d}".format(frame_idx_0)+".png") and os.path.exists(    "./labeled_imgs/"+ main_folder + "/hd_foreground_134_"+"{:03d}".format(frame_idx_0)+".png"):
        print("jump existed")
        return
    frame_idx_1 = frame_idx_0 + frame_predict_step
    if input_type == "img":
        frame_idx_0_path = png_path_prefix + "."+  "{:03d}".format(frame_idx_0 + 1) + ".png"
        frame_idx_1_path = png_path_prefix + "."+  "{:03d}".format(frame_idx_1 + 1) + ".png"
        frame_n0 = cv2.imread(frame_idx_0_path)
        frame_n1 = cv2.imread(frame_idx_1_path)
    elif input_type == "vid":
        frame_n0 = frames[frame_idx_0]
        frame_n1 = frames[frame_idx_1]

    # ratio = len(frame_n0)/len(frame_n0[0]) # h/ w
    # set_h = 512
    # frame_n0 = cv2.resize(frame_n0, (int(set_h/ratio), set_h), interpolation=cv2.INTER_LINEAR)
    # frame_n1 = cv2.resize(frame_n1, (int(set_h/ratio), set_h), interpolation=cv2.INTER_LINEAR)
    # if frame_n0[0] > 600:
    #     frame_n0 = cv2.resize(frame_n0, (len(frame_n0[0])//2, len(frame_n0)//2), interpolation=cv2.INTER_LINEAR)
    #     frame_n1 = cv2.resize(frame_n1, (len(frame_n1[0])//2, len(frame_n1)//2), interpolation=cv2.INTER_LINEAR)
    # frame_n0 = cv2.resize(frame_n0, (block_size*64, block_size*36), interpolation=cv2.INTER_LINEAR)
    # frame_n1 = cv2.resize(frame_n1, (block_size*64, block_size*36), interpolation=cv2.INTER_LINEAR)
    # preprocess_a_frame_size_inplace(frame_n0, block_size)

    frame_n0 = preprocess_a_frame_size(frame_n0, block_size, resizeT_cutoutF=resizeT_cutoutF) # here is blockized
    frame_n1 = preprocess_a_frame_size(frame_n1, block_size, resizeT_cutoutF=resizeT_cutoutF) # here is blockized
    frame_n0_Y_blockized = frame_n0
    frame_n1_Y_blockized = frame_n1
    # frame_n0_Y_blockized = preprocess_a_frame_to_Y(frame_n0, block_size)
    # frame_n1_Y_blockized = preprocess_a_frame_to_Y(frame_n1, block_size)
    frame_n0_Y_blockized = preprocess_a_frame_to_HSV(frame_n0, block_size)
    frame_n1_Y_blockized = preprocess_a_frame_to_HSV(frame_n1, block_size)
    cur_vector_file_name = "motion_vector"+"{:03d}".format(frame_idx_0)
    cur_vector_file_path = motion_vector_storage + "/" + cur_vector_file_name
    if os.path.exists(cur_vector_file_path):
        print("loading motion vector" + str(frame_idx_0))
        # load data
        storage_file = open(cur_vector_file_path, 'rb')
        motion_vector_matrix = pickle.load(storage_file)
        storage_file.close()
    else:
        print("calculating motion vector" + str(frame_idx_0))
        motion_vector_matrix, predicted = get_motion_vector_matrix(frame_n0_Y_blockized, frame_n1_Y_blockized, block_size, search_method, search_expand_length, max_best_candidates_per_level, if_calculate_prediction_and_output)
        if if_calculate_prediction_and_output:
            draw_line_on_predicted(predicted, motion_vector_matrix, block_size)
            save_intermediate_images(frame_n1_Y_blockized, predicted, idx = frame_idx_0)
        motion_vector_candidates_set_min_length_to_candidate_idx_0_inplace(motion_vector_matrix)
        # store
        storage_file = open(cur_vector_file_path, 'wb')
        pickle.dump(motion_vector_matrix, storage_file)
        storage_file.close()

    # cutting out four edges
    motion_vector_matrix_h = len(motion_vector_matrix)
    motion_vector_matrix_w = len(motion_vector_matrix[0])
    motion_vector_matrix = np.array([motion_vector_matrix[x][1:motion_vector_matrix_w-1] for x in range(1,motion_vector_matrix_h-1)])
    #                here is at [x,y] map to 1d clustered labels
    cluster_data, motion_vector_at_xy_to_idx = make_cluster_dataset_of_motion_vectors(motion_vector_matrix)
    #motion_vector_matrix = motion_vector_matrix[1:motion_vector_matrix_h-1,1:motion_vector_matrix_w-1,:,:]
    frame_n1_h = len(frame_n1)
    frame_n1_w = len(frame_n1[0])
    frame_n1 = np.array([frame_n1[x][block_size:frame_n1_w-block_size] for x in range(block_size,frame_n1_h-block_size)])
    

    # just for print
    # dydx_tuples = []
    # for h in range(len(motion_vector_matrix)):
    #     for w in range(len(motion_vector_matrix[0])):
    #         dydx_tuples.append(tuple(motion_vector_matrix[h][w][0]))
    # print(Counter(dydx_tuples))
    # print(cluster_data)
    use_threshold_splitting = True
    if use_threshold_splitting:
        ##############
        # threshold splitting
        
        # print("threshold splitting")
        count_on_dy_dx = collections.defaultdict(int)
        max_count = 0
        a_block_y_x_of_max_count = None

        motion_vector_matrix_h = len(motion_vector_matrix)
        motion_vector_matrix_w = len(motion_vector_matrix[0])

        dydx_tuples = []
        for h in range(motion_vector_matrix_h):
            for w in range(motion_vector_matrix_w):
                dydx_tuples.append((int(motion_vector_matrix[h][w][0][0]),int(motion_vector_matrix[h][w][0][1])))
        dydx_tuples_counts = Counter(dydx_tuples) # (dy,dx): count
        descending_dydx_tuples = sorted(dydx_tuples_counts, key=dydx_tuples_counts.get, reverse=True)
        # print(descending_dydx_tuples)

        for h in range(motion_vector_matrix_h):  
            for w in range(motion_vector_matrix_w):
                cur_dy_dx = (motion_vector_matrix[h][w][0][0], motion_vector_matrix[h][w][0][1])
                count_on_dy_dx[cur_dy_dx] += 1
                if count_on_dy_dx[cur_dy_dx] > max_count:
                    max_count = count_on_dy_dx[cur_dy_dx]
                    a_block_y_x_of_max_count = cur_dy_dx

        search_start_candidate_number = 0
        for i in range(len(descending_dydx_tuples)):
            if dydx_tuples_counts[descending_dydx_tuples[i]] < over_this_threshold_count_set_as_search_start_bkg_threshold:
                break
            else:
                search_start_candidate_number = i

        search_start_blocks_y_x = []
        for h in range(motion_vector_matrix_h):  
            for w in range(motion_vector_matrix_w):
                for ith_most_count_dy_dx in range(search_start_candidate_number):
                    if int(motion_vector_matrix[h][w][0][0]) == descending_dydx_tuples[ith_most_count_dy_dx][0] and int(motion_vector_matrix[h][w][0][1]) == descending_dydx_tuples[ith_most_count_dy_dx][1]:
                        search_start_blocks_y_x.append((h,w))
                        break
        
        
        # a_block_x_y_of_max_count is now search start as background, 2 not searched, 1 is obj, 0 is bkg
        blocks_class_mask = [[2 for i in range(motion_vector_matrix_w)] for j in range(motion_vector_matrix_h)]
        for start_y_x in search_start_blocks_y_x:
            blocks_class_mask[start_y_x[0]][start_y_x[1]] = 0

        # bfs search all the unsearched neigibours, if over tolerant range set as object
        q = collections.deque(search_start_blocks_y_x)
        while q:
            cur_i, cur_j = q.popleft()
            for direction in motion_difference_threshold_search_directions:
                nei_i = cur_i + direction[0]
                nei_j = cur_j + direction[1]
                if 0 <= nei_i < motion_vector_matrix_h and 0 <= nei_j < motion_vector_matrix_w:
                    if blocks_class_mask[nei_i][nei_j] == 2:# not searched
                        is_in_range = True
                        for i in range(len(motion_difference_tolerate_thresholds)):
                            threshold = motion_difference_tolerate_thresholds[i]
                            if abs(motion_vector_matrix[cur_i][cur_j][0][i] - motion_vector_matrix[nei_i][nei_j][0][i]) > threshold:
                                # out of range
                                is_in_range = False
                                blocks_class_mask[nei_i][nei_j] = 1
                                break
                        if is_in_range:
                            # q.append((nei_i, nei_j))
                            blocks_class_mask[nei_i][nei_j] = 0



        # check all not searched if so steep from others
        # vote from neighbours
        #                     bkg    obj
        #            successive vote
        mask_label_sets = [set([0,3]), set([1,4])]
        for cur_i in range(motion_vector_matrix_h):
            for cur_j in range(motion_vector_matrix_w):
                if blocks_class_mask[cur_i][cur_j] == 2:# not searched
                    neighbour_num = 0
                    nei_is_obj_vote = 0
                    for direction in motion_difference_threshold_search_directions:
                        nei_i = cur_i + direction[0]
                        nei_j = cur_j + direction[1]
                        if 0 <= nei_i < motion_vector_matrix_h and 0 <= nei_j < motion_vector_matrix_w:
                            neighbour_num += 1 # nei exist
                            if blocks_class_mask[nei_i][nei_j] in mask_label_sets[0]:
                                nei_is_obj_vote += 1
                    if nei_is_obj_vote > neighbour_num/2:
                        blocks_class_mask[cur_i][cur_j] = 4
                    else:
                        blocks_class_mask[cur_i][cur_j] = 3
        pass



    using_clustering = False
    if using_clustering:
        ##############
        # clustering

        #motion_vector_matrix = motion_vector_matrix[1:motion_vector_matrix_h-1,1:motion_vector_matrix_w-1,:,:]
        frame_n0_h = len(frame_n0)
        frame_n0_w = len(frame_n0[0])
        frame_n1_h = len(frame_n1)
        frame_n1_w = len(frame_n1[0])
        frame_n0=np.array([frame_n0[x][block_size:frame_n0_w-block_size] for x in range(block_size,frame_n0_h-block_size)])
        frame_n1 = np.array([frame_n1[x][block_size:frame_n1_w-block_size] for x in range(block_size,frame_n1_h-block_size)])
        #frame_n1=frame_n1[block_size:frame_n1_h-block_size,block_size:frame_n1_w-block_size]
        cluster_labels = cluster(cluster_data)

        labels_descending = get_cluster_labels_descending(cluster_labels)
        clustered_labels_to_ascending_labels_by_count = dict()
        for idx in range(len(labels_descending)):
            clustered_labels_to_ascending_labels_by_count[labels_descending[idx][0]] = idx
        print(len(labels_descending), "clusters")

        # mark block belongs to which class, class ascending from 0
        blocks_class_mask = [[0 for i in range(len(motion_vector_matrix[0]))] for j in range(len(motion_vector_matrix))]
        for h in range(len(motion_vector_matrix)):
            for w in range(len(motion_vector_matrix[0])):
                blocks_class_mask[h][w] = clustered_labels_to_ascending_labels_by_count[cluster_labels[motion_vector_at_xy_to_idx[(h, w)]]]


        frame_n1_RGBA_base = cv2.cvtColor(frame_n1, cv2.COLOR_RGB2RGBA)
        extracted_frame_n1_RGBA_s = []
        for label, label_count in labels_descending:
            frame_n1_RGBA = frame_n1_RGBA_base.copy()
            for w in range(frame_n1_RGBA.shape[1]):
                for h in range(frame_n1_RGBA.shape[0]):
                    h_idx = h // block_size
                    w_idx = w // block_size
                    cur_pixel_belongs_to_block_idx = motion_vector_at_xy_to_idx[(h_idx, w_idx)]
                    if cluster_labels[cur_pixel_belongs_to_block_idx] == label:
                        frame_n1_RGBA[h][w][3] = 255
                    else:
                        frame_n1_RGBA[h][w] = [0, 0, 0, 0]
            extracted_frame_n1_RGBA_s.append(frame_n1_RGBA)
            # cv2.imshow("frame_n1_RGBA" + str(label), frame_n1_RGBA)
            # cv2.waitKey(0)
        
        # cv2.imshow("frame_n1_RGBA" + str(0), extracted_frame_n1_RGBA_s[0])
        # cv2.waitKey(0)

    # ps=[]
    # ds=[]
    # #for borderSize in range(0,10):
    # for borderSize in range(3,4):
    #     try:
    #         p, d = getFourPoints(motion_vector_matrix,extracted_frame_n1_RGBA_s,block_size,borderSize,centerPoint)
    #         #print(p,d)
    #         #d = [[item[0]+block_size,item[1]+block_size] for item in d]
    #         ps.append(p)
    #         ds.append(d)
    #     except:
    #         pass      
    # #print(extracted_frame_n1_RGBA_s[0].shape[:2])
    # #print(frame_n0.shape[:2])
    # if(wholePanorama.any()==False):
    #     wholePanorama=np.copy(extracted_frame_n1_RGBA_s[0])
    # else:
    #     wholePanorama,centerPoint = generatePanoramaCandidate(extracted_frame_n1_RGBA_s[0],ps,wholePanorama,ds)
    # cv2.namedWindow("wholepanorama"+str(frame_idx_1), cv2.WINDOW_NORMAL) 
    # cv2.imshow("wholepanorama"+str(frame_idx_1),wholePanorama)
    # cv2.imwrite("./paromaraOutput/"+"wholepanorama"+str(frame_idx_1)+".png",wholePanorama)
    # cv2.imwrite("./paromaraOutput/"+"background"+str(frame_idx_1)+".png",extracted_frame_n1_RGBA_s[0])
    # cv2.waitKey(0)
    # #print(centerPoint)
    # #print(1)
    # pass

    if labeled_generation_mode == "hd":
        all_resolution_frame = all_resolution_frames[frame_idx_1]
        # cut edges
        ratio_h = len(all_resolution_frame)/len(frame_n1)
        ratio_w = len(all_resolution_frame[0])/len(frame_n1[0])
        all_resolution_frame = np.array([all_resolution_frame[x][int(block_size * ratio_w):len(all_resolution_frame[0])-int(block_size * ratio_w)] for x in range(int(block_size * ratio_h),len(all_resolution_frame)-int(block_size * ratio_h))])
        ratio_h = len(all_resolution_frame)/len(frame_n1)
        ratio_w = len(all_resolution_frame[0])/len(frame_n1[0])
        frame_n1_RGBA_base = cv2.cvtColor(all_resolution_frame, cv2.COLOR_RGB2RGBA)
        extracted_frame_n1_RGBA_s = []
        for mask_label_set in [set([0]),set([1,3,4])]:
        # for mask_label_set in mask_label_sets:
            frame_n1_RGBA = frame_n1_RGBA_base.copy()
            for w in range(frame_n1_RGBA.shape[1]):
                for h in range(frame_n1_RGBA.shape[0]):
                    h_idx = int(h / (block_size * ratio_h))
                    w_idx = int(w / (block_size * ratio_w))
                    if blocks_class_mask[h_idx][w_idx] in mask_label_set:
                        frame_n1_RGBA[h][w][3] = 255
                    else:
                        frame_n1_RGBA[h][w][3] = 0
            extracted_frame_n1_RGBA_s.append(frame_n1_RGBA)
            # cv2.imshow("frame_n1_RGBA" + str(mask_label_set), frame_n1_RGBA)
        # cv2.waitKey(0)
        cv2.imwrite("./labeled_imgs/"+ main_folder + "/hd_background_0_"+"{:03d}".format(frame_idx_0)+".png", extracted_frame_n1_RGBA_s[0])
        cv2.imwrite("./labeled_imgs/"+ main_folder + "/hd_foreground_134_"+"{:03d}".format(frame_idx_0)+".png", extracted_frame_n1_RGBA_s[1])
    elif labeled_generation_mode == "c":
        frame_n1_RGBA_base = cv2.cvtColor(frame_n1, cv2.COLOR_RGB2RGBA)
        extracted_frame_n1_RGBA_s = []
        for mask_label_set in [set([0]),set([1,3,4])]:
        # for mask_label_set in mask_label_sets:
            frame_n1_RGBA = frame_n1_RGBA_base.copy()
            for w in range(frame_n1_RGBA.shape[1]):
                for h in range(frame_n1_RGBA.shape[0]):
                    h_idx = h // block_size
                    w_idx = w // block_size
                    if blocks_class_mask[h_idx][w_idx] in mask_label_set:
                        frame_n1_RGBA[h][w][3] = 255
                    else:
                        frame_n1_RGBA[h][w][3] = 0
            extracted_frame_n1_RGBA_s.append(frame_n1_RGBA)
            # cv2.imshow("frame_n1_RGBA" + str(mask_label_set), frame_n1_RGBA)
        # cv2.waitKey(0)
        if not os.path.exists("./labeled_imgs/"+ main_folder):
            os.mkdir("./labeled_imgs")
            os.mkdir("./labeled_imgs/"+ main_folder)
        cv2.imwrite("./labeled_imgs/"+ main_folder + "/background_0_"+"{:03d}".format(frame_idx_0)+".png", extracted_frame_n1_RGBA_s[0])
        cv2.imwrite("./labeled_imgs/"+ main_folder + "/foreground_134_"+"{:03d}".format(frame_idx_0)+".png", extracted_frame_n1_RGBA_s[1])

    print("finish", frame_idx_0)
            

if __name__ == "__main__":
    # test.py arg1 arg2
    ##############

    input_type = "vid"

    png_path_prefix = None
    frames = None
    all_resolution_frames = None
    if input_type == "img":
        # image
        if len(sys.argv) == 2:
            directory = sys.argv[1]
            # sys.argv[2]
        else:
            # directory = "C:/Users/14048/Desktop/multimedia/project/video_rgb/SAL_490_270_437"
            directory = "C:\\Users\\14048\\Desktop\\multimedia\\project\\video_rgb\\Stairs_490_270_346"
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

        if not os.path.exists(parent_dict+"/"+main_folder+"/"+main_folder+".001.png"):
            runJavaRGB2PNG(parent_dict+"/", main_folder)

    elif input_type == "vid":
        ##############
        video_path = "../video_view/Stairs_compact.mp4"
        video_high_resolution_path = "../video_view/Stairs.mp4"
        # video_path = "D:\\chrome downloads\\final_demo_data\\final_demo_data/test2.mp4"
        # video_path = "/Users/piaomz/Desktop/CSCI576/final_demo_data/test1.mp4"
        splitted1 = video_path.split("/")
        splitted2 = []
        for splitted1_seg in splitted1:
            splitted2 += splitted1_seg.split("\\")
        main_folder = splitted2[-1].split(".")[0]
        frames, fps = convert_video_2_bgra(video_path)
        frame_num = len(frames)


    if_calculate_prediction_and_output = True

    use_multiprocessing = False

    # if generate with high resolution, resizeT_cutoutF must be True, resized then calculatr motion vector
    # "hd" generate hd
    # "c" generate compacted
    # else pass
    labeled_generation_mode = "c"
    if labeled_generation_mode == "hd":
        all_resolution_frames, fps_dummy = convert_video_2_bgra(video_high_resolution_path)

    block_size = 16
    search_expand_length = 16
    frame_predict_step = 20
    max_best_candidates_per_level = 10
    # block_size = 32
    # search_expand_length = 32
    # frame_predict_step = 4
    resizeT_cutoutF = True
    search_method = "l"  # h for hierarchical b for brute force l for lucas
                            # if lucas, frame_predict_step need to be 1

    motion_difference_tolerate_thresholds = [3, 3, 1000]
    over_this_threshold_count_set_as_search_start_bkg_threshold = 10
    motion_difference_threshold_search_directions = [(1,0),(0,1),(-1,0),(0,-1)]

    folder_prefix = main_folder + "_" \
        + str(block_size) + "_" \
            + str(search_expand_length) + "_" \
                + str(frame_predict_step) + "_" \
                    + str(max_best_candidates_per_level) + "_" \
                        + str(resizeT_cutoutF) + "_" \
                            + search_method + "_" \
                                + str(motion_difference_tolerate_thresholds) + "_" \
                                    + str(over_this_threshold_count_set_as_search_start_bkg_threshold) + "_" \
                                        + str(motion_difference_threshold_search_directions)
    motion_vector_storage = "./motion_vector_storage/" + folder_prefix
    if not os.path.exists(motion_vector_storage):
        os.makedirs(motion_vector_storage)
        print("calculate from start", folder_prefix)
    else:
        print("some motion vector calculated, take and store in ", folder_prefix)
    if not os.path.exists("./labeled_imgs"):
        os.mkdir("./labeled_imgs")
    if not os.path.exists("./labeled_imgs/"+ main_folder):
        os.mkdir("./labeled_imgs/"+ main_folder)

    wholePanorama=np.array([])
    centerPoint=[0,0]

    print("all", frame_num-frame_predict_step, "iterations")
    if use_multiprocessing:
        print("using parallel processing (CPU)")
        pool = multiprocessing.Pool(processes=24)
        parameters_each_iter = []
    # for frame_idx_0 in range(frame_num-frame_predict_step):
    # for frame_idx_0 in range(0, frame_num-frame_predict_step, frame_predict_step):
        # parameters_each_iter.append([frame_idx_0, frame_predict_step, input_type, png_path_prefix, frames, resizeT_cutoutF, motion_vector_storage, over_this_threshold_count_set_as_search_start_bkg_threshold, search_method, max_best_candidates_per_level, motion_difference_threshold_search_directions,motion_difference_tolerate_thresholds, main_folder])
        pool.map(partial(main_loop, frame_predict_step = frame_predict_step, input_type = input_type, png_path_prefix = png_path_prefix, frames = frames, resizeT_cutoutF = resizeT_cutoutF, motion_vector_storage = motion_vector_storage, over_this_threshold_count_set_as_search_start_bkg_threshold = over_this_threshold_count_set_as_search_start_bkg_threshold, search_method = search_method, max_best_candidates_per_level = max_best_candidates_per_level, motion_difference_threshold_search_directions = motion_difference_threshold_search_directions,motion_difference_tolerate_thresholds = motion_difference_tolerate_thresholds, main_folder = main_folder, block_size = block_size, search_expand_length=search_expand_length, if_calculate_prediction_and_output = if_calculate_prediction_and_output, all_resolution_frames = all_resolution_frames, labeled_generation_mode = labeled_generation_mode), [frame_idx_0 for frame_idx_0 in range(frame_num-frame_predict_step)])
    else:
        for frame_idx_0 in range(frame_num-frame_predict_step):
            main_loop(frame_idx_0, frame_predict_step, input_type, png_path_prefix, frames, resizeT_cutoutF, motion_vector_storage, over_this_threshold_count_set_as_search_start_bkg_threshold, search_method, max_best_candidates_per_level, motion_difference_threshold_search_directions,motion_difference_tolerate_thresholds, main_folder, block_size, search_expand_length, if_calculate_prediction_and_output, all_resolution_frames, labeled_generation_mode)

