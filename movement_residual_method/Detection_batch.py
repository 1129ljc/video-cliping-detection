'''
@Time    : 2021/4/25 15:58
@Author  : ljc
@FileName: Detection_batch.py
@Software: PyCharm
'''
import sys

sys.path.append("../")
import os
import datetime
from movement_residual_method.Feature_Deal import detection
from multiprocessing import Process


def detection_batch(video_fake, detection_out_save, txt_info_save):
    names = os.listdir(video_fake)
    txt_info = open(file=txt_info_save, mode='w', encoding='utf-8')
    txt_info.write('运动残差阈值:0.4 波动强度阈值:1.6\n')

    for i in range(len(names)):
        # 获取待检测视频
        name = names[i]
        video = video_fake + '/' + name
        detection_out_save_save = detection_out_save
        result = detection(video, detection_out_save_save)
        result_str = ''
        for idx in range(1, len(result)):
            result_str = result_str + str(result[idx]) + '\t'
        info_str = '{video_name}\t{result_str}\n'.format(
            video_name=name + (32 + 4 - len(name)) * ' ',
            result_str=result_str
        )
        txt_info.write(info_str)
    txt_info.close()
    print(video_fake, 'done!')


if __name__ == '__main__':
    video_fake_dir = 'Z:/ljc/video_dataset_frame/UCF-101_frame_del_1/'
    detection_out_dir = 'Z:/ljc/video_dataset_frame/frame_del_detection_by_motion_residual_1/'
    txt_info = 'Z:/ljc/video_dataset_frame/info_del_detection_by_motion_residual_1/'
    names = os.listdir(video_fake_dir)
    names = names[3:]
    print(names)
    for i in range(len(names)):
        name = names[i]
        video_fake = os.path.join(video_fake_dir, name)
        detection_out_save = os.path.join(detection_out_dir, name)
        txt_info_save = os.path.join(txt_info, name + '.txt')
        print(video_fake)
        print(detection_out_save)
        print(txt_info_save)
        detection_batch(video_fake, detection_out_save, txt_info_save)

        # process = Process(target=detection_batch, name=video_fake, args=(video_fake, detection_out_save, txt_info_save))
        # print('进程开启：', video_fake)
        # process.start()
