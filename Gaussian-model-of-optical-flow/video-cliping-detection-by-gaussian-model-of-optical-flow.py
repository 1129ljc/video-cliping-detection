import os
import cv2
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector


def find_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    return scene_manager.get_scene_list()


def optical_flow_extract(frame1, frame2):
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 5, 13, 10, 5, 1.1, 0)
    flow_x = np.abs(flow[..., 0])
    flow_y = np.abs(flow[..., 1])
    mag = np.sqrt(np.add(np.power(flow_x, 2), np.power(flow_y, 2)))
    mag = np.abs(np.reshape(mag, (1, -1)))
    OF = np.sum(mag, dtype=np.float64)
    return OF


def show(data):
    x = [i for i in range(len(data))]
    y = data
    plt.bar(x, y)
    plt.title("gaussian optical flow")
    plt.xlabel("frame sequence")
    return plt


def featrue_extract(input_file, start, end):
    print(input_file)
    cap = cv2.VideoCapture(input_file)
    flow_size = int(end - start)
    OF = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(start, end):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        flow = optical_flow_extract(frame1, frame2)
        OF.append(flow)
    cap.release()
    variation_factor = []
    for i in range(len(OF)):
        x = 0
        if i == 0:
            x = (2 * OF[i]) / (OF[i + 1] + OF[i + 2])
        elif i > 0 and i < len(OF) - 1:
            x = (2 * OF[i]) / (OF[i - 1] + OF[i + 1])
        elif i == len(OF) - 1:
            x = (2 * OF[i]) / (OF[i - 1] + OF[i - 2])
        variation_factor.append(x)
    return variation_factor


def z_gaussian_model(variation_factor):
    temp = np.array(variation_factor, dtype=np.float64)
    mean = np.mean(temp)
    var = np.var(temp)
    z = np.abs(temp - mean)
    z = np.true_divide(z, var)
    return z.tolist()


def judge_threshold(z, T_del):
    deletion_position = []
    for i in range(len(z)):
        if z[i] >= T_del:
            deletion_position.append(i)
    return deletion_position


def gen_abnormal_point_pic(video, deletion_position):
    video_cap = cv2.VideoCapture(video)
    frame_num = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    pics = []
    for i in range(len(deletion_position)):
        location = deletion_position[i]
        if location < 5:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif location > frame_num - 5:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 10)
        else:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        images = []
        for idx in range(10):
            ref, frame = video_cap.read()
            images.append(frame)
        image = np.concatenate((np.concatenate((images[0:5]), axis=1), np.concatenate((images[5:10]), axis=1)), axis=0)
        pics.append(image)
    return pics, fps


def detection(video, start, end):
    variation_factor = featrue_extract(video, start, end)
    z = z_gaussian_model(variation_factor)
    deletion_position = judge_threshold(z, 5)
    plt = show(z)
    return deletion_position, plt


def main(input_a, input_b, input_c):
    task_id = input_a
    file_path = input_b
    ext = input_c

    input_arg_ext_json = json.loads(ext)
    input_arg_ext_out_json_path = input_arg_ext_json['JSON_FILE_PATH']

    input_arg_ext_tmp_dir = input_arg_ext_json['TMP_DIR']
    input_arg_ext_tmp_dir = os.path.join(input_arg_ext_tmp_dir, 'ljc_docs')

    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_tmp_dir, 'detection_by_gaussian_optical_flow')
    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_out_tmp_path, str(task_id))

    if not os.path.exists(input_arg_ext_out_tmp_path):
        os.makedirs(input_arg_ext_out_tmp_path)

    video_names = os.listdir(file_path)
    algorithm_message = '该算法计算相邻两帧的光流，计算光流梯度值，并使用一维高斯模型进行建模，阈值法筛选异常点。'

    result_json_content = {}
    for index in range(len(video_names)):
        video_name_detec = video_names[index]
        video_name_path = os.path.join(file_path, video_name_detec)
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
                                    'title': '相邻帧光流梯度变换特征高斯模型特征直方图',
                                    'comment': '左侧表示检测视频帧编号在' + str(start) + '-' + str(end) + '时间范围内的邻帧间光流梯度变换特征，使用高斯模型进行建模，展示建模后的特征异常量化直方图。'}
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
        video_json = {'taskid': str(task_id),
                      'conclusion': video_conclusion,
                      'message': algorithm_message,
                      'confidence': confidence_all,
                      'threshold': 0,
                      'features': video_features}
        result_json_content[str(video_name_detec)] = video_json
        print("done")

    with open(input_arg_ext_out_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json_content, f, ensure_ascii=False)
    f.close()


if __name__ == '__main__':
    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    input_3 = sys.argv[3]
    main(input_a=input_1, input_b=input_2, input_c=input_3)
