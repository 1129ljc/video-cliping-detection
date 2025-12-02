import os
import json
import sys
import cv2
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def find_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    return scene_manager.get_scene_list()


def detection(video_path, start, end):
    video_cap = cv2.VideoCapture(video_path)
    frame_num = int(end - start)
    winSize = (128, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    winStride = (8, 8)
    feature = np.zeros(shape=[frame_num, 8100])
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(frame_num):
        ref, frame = video_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (128, 128))
        test_hog = hog.compute(frame, winStride)
        # print(test_hog.shape, feature.shape)
        feature[i, :] = test_hog.squeeze()
    C = np.zeros(shape=[frame_num - 1])
    for index in range(frame_num - 1):
        i, j = index, index + 1
        VF_i, VF_j = feature[i, :], feature[j, :]
        u_i, u_j = np.mean(VF_i), np.mean(VF_j)
        fenzi = np.sum(np.multiply(VF_i - u_i, VF_j - u_j))
        fenmu = np.sqrt(np.sum(np.multiply(np.multiply(VF_i - u_i, VF_i - u_i), np.multiply(VF_j - u_j, VF_j - u_j))))
        C[index] = fenzi / fenmu
    C_mean = np.mean(C)
    C_std = np.std(C)
    C = np.abs(C - C_mean) / C_std
    n = int(C.shape[0])
    interval = stats.t.interval(0.99, n - 2, loc=C.mean(), scale=C.std(ddof=1))
    t = np.max(interval)
    G = ((n - 1) * t) / (np.sqrt(n * (n - 2 + t * t)))
    judge = []
    for i in range(n):
        if C[i] > G:
            judge.append(i)
    plt.plot([i for i in range(C.shape[0])], C)
    plt.title("插帧样本相邻帧HOG特征相关性折线")
    plt.xlabel("视频序列帧")
    plt.ylabel("特征相关性量化幅度值")
    return judge, plt


def test_videos(input_a, input_b, input_c):
    input_arg_task_id = input_a
    input_arg_file_path = input_b
    input_arg_ext = input_c

    input_arg_ext_json = json.loads(input_arg_ext)
    input_arg_ext_out_json_path = input_arg_ext_json['JSON_FILE_PATH']

    input_arg_ext_tmp_dir = input_arg_ext_json['TMP_DIR']
    input_arg_ext_tmp_dir = os.path.join(input_arg_ext_tmp_dir, 'ljc_docs')

    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_tmp_dir, 'detection_hog_feature')
    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_out_tmp_path, str(input_arg_task_id))

    if not os.path.exists(input_arg_ext_out_tmp_path):
        os.makedirs(input_arg_ext_out_tmp_path)

    algorithm_message = '该算法针对视频样本逐帧提取8100维度HOG特征，计算该特征相邻帧的相关性，使用高斯模型建模，采用T分布自适应计算模型阈值，超出阈值则为异常点。'
    print(algorithm_message)

    files = os.listdir(input_arg_file_path)
    result_json_content = {}
    for i in range(len(files)):
        video_name_detec = files[i]
        video_name_path = os.path.join(input_arg_file_path, video_name_detec)
        scenes = find_scenes(video_name_path)
        video_features = []
        confidence = []
        confidence_all = 0
        for scene_index in range(len(scenes)):
            scene = scenes[scene_index]
            start, end = scene[0].frame_num + 2, scene[1].frame_num - 2
            print('Analyze paragraph ' + str(scene_index + 1) + ' Start: ' + str(start) + ' ' + 'End: ' + str(end))
            deletion_position, plt = detection(video_name_path, start, end)
            plt_save = os.path.join(input_arg_ext_out_tmp_path, video_name_detec.split('.')[0] + '_' + str(scene_index) + '.jpg')
            plt.savefig(plt_save)
            plt.close()
            confidence_scene_index = 0
            video_feature_common = {'filepath': plt_save,
                                    'title': '相邻帧HOG特征相关性折线图',
                                    'comment': '左侧表示检测视频帧编号在' + str(start) + '-' + str(end) + '时间范围内的相邻帧HOG特征相关性折线图。'}
            video_feature_common['comment'] = video_feature_common['comment'] + '折线图中若存在明显的峰值特征，则对应峰值横坐标帧编号即为该段内的异常点'
            video_feature_common['comment'] = video_feature_common['comment'] + '该视频片段是镜头序列中第' + str(scene_index + 1) + '段，累计' + str(len(scenes)) + '段。'
            if len(deletion_position) != 1 and len(deletion_position) != 2:
                video_feature_common['comment'] = video_feature_common['comment'] + '该视频片段为真实视频。'
                confidence_scene_index = 0
            else:
                confidence_scene_index = 1
                if len(deletion_position) == 1:
                    video_feature_common['comment'] = video_feature_common['comment'] + '该视频片段是删帧视频; 删帧点帧编号 : ' + str(start + deletion_position[0]) + '。'
                elif len(deletion_position) == 2:
                    start_1 = min(int(deletion_position[0]), int(deletion_position[1]))
                    end_1 = max(int(deletion_position[0]), int(deletion_position[1]))
                    video_feature_common['comment'] = video_feature_common['comment'] + '该视频片段是插帧视频; 插帧开始点帧编号 : ' + str(start + start_1) + '; 插帧结束点帧编号 : ' + str(start + end_1) + '。'
            video_features.append(video_feature_common)
            confidence.append(confidence_scene_index)

        judge = np.array(confidence)
        video_conclusion = ''
        if np.any(judge == 1):
            location = np.where(judge == 1)[0]
            print(location)
            conclusion_s = ''
            for location_index in range(int(location.shape[0])):
                conclusion_s = conclusion_s + str(location[location_index] + 1) + '、'
            conclusion_s = conclusion_s[0:-1]
            video_conclusion = '该视频按镜头切分为' + str(len(scenes)) + '段；该视频为伪造视频；可疑点存在于第' + conclusion_s + '段。'
            confidence_all = 1
        elif np.all(judge == 0):
            video_conclusion = '该视频按镜头切分为' + str(len(scenes)) + '段；所有段均为真实视频。'
            confidence_all = 0
        video_json = {'taskid': str(input_arg_task_id),
                      'conclusion': video_conclusion,
                      'message': algorithm_message,
                      'confidence': confidence_all,
                      'threshold': 0,
                      'features': video_features}
        result_json_content[str(video_name_detec)] = video_json
        print("done")

    with open(input_arg_ext_out_json_path, 'w') as f:
        json.dump(result_json_content, f)
    f.close()


if __name__ == '__main__':
    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    input_3 = sys.argv[3]
    test_videos(input_a=input_1, input_b=input_2, input_c=input_3)
