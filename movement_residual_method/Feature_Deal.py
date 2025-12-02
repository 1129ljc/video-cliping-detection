import numpy as np
import time
import matplotlib.pyplot as plt
import os
import cv2


def Video_print_and_init(Input_file):
    '''
    打印并初始化视频基本信息
    :param Input_file: 视频文件
    :return: 无
    '''
    capture = cv2.VideoCapture(Input_file)
    frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("视频帧数量：{frame_num}".format(frame_num=frame_num))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print("视频宽高:({frame_width}x{frame_height})".format(frame_width=frame_width, frame_height=frame_height))
    return frame_num, frame_width, frame_height


def Video_to_YUV(input_file, YUV_file):
    '''
    将视频转成无压缩的YUV格式
    :param input_file:输入文件，例如【input_file。mp4】
    :param YUV_file: 输出文件，例如【YUV_file.yuv】
    :return: 无
    '''
    cmd = 'ffmpeg -loglevel quiet -i {input} {output}'.format(input=input_file, output=YUV_file)
    # print("待检测视频解码成YUV序列")
    t1 = time.time()
    # print(cmd)
    if os.path.exists(YUV_file):
        os.system("del {filename}".format(filename=YUV_file))
        os.system(cmd)
    else:
        os.system(cmd)
    t2 = time.time()
    # print("将视频解码为YUV序列时间消耗%.3f秒" % (t2 - t1))
    return 0


def Encoder_first_by_newx264(YUV_file, H264_file1, frame_width, frame_height):
    '''
    使用编码器x264_gop_250提取特征，GOP设置为250，提取P帧的逐像素运动残差特征
    :param YUV_file: 输入文件
    :param H264_file1:编码结果文件【不使用】
    :return:无
    '''
    # print("使用改进编码器对YUV序列进行首次编码得到feature_gop_250特征文件")
    t1 = time.time()
    feature_file = "feature_gop_250.txt"
    shell_delete = "del {file}".format(file=feature_file)
    shell_encoder = "x264_gop_250.exe --profile baseline --fps 30 --threads 1 --no-scenecut --keyint 250 --input-res {frame_width}x{frame_height} -o {H264_file} {YUV_file}".format(
        frame_width=frame_width, frame_height=frame_height, H264_file=H264_file1, YUV_file=YUV_file)
    shell_encoder = shell_encoder + ' 2> Z:/ljc/frame_tamper_code/movement_residual_method/output1.txt'
    # print(shell_encoder)
    if os.path.exists(feature_file):
        os.system("{}&&{}".format(shell_delete, shell_encoder))
    else:
        os.system(shell_encoder)
    t2 = time.time()
    # print("首次编码YUV序列时间消耗%.3f秒" % (t2 - t1))
    return 0


def Encoder_second_by_newx264(YUV_file, H264_file2, frame_width, frame_height):
    '''
    使用编码器x264_gop_200提取特征，GOP设置为200，提取P帧的逐像素运动残差特征
    :param YUV_file: 输入文件
    :param H264_file2:编码结果文件【不使用】
    :return:无
    '''
    # print("使用改进编码器对YUV序列进行二次编码得到feature_gop_200特征文件")
    t1 = time.time()
    feature_file = "feature_gop_200.txt"
    shell_delete = "del {file}".format(file=feature_file)
    shell_encoder = "x264_gop_200.exe --profile baseline --fps 30 --threads 1 --no-scenecut --keyint 200 --input-res {frame_width}x{frame_height} -o {H264_file} {YUV_file}".format(
        frame_width=frame_width, frame_height=frame_height, H264_file=H264_file2, YUV_file=YUV_file)
    shell_encoder = shell_encoder + ' 2> Z:/ljc/frame_tamper_code/movement_residual_method/output2.txt'
    # print(shell_encoder)
    if os.path.exists(feature_file):
        os.system("{}&&{}".format(shell_delete, shell_encoder))
    else:
        os.system(shell_encoder)
    t2 = time.time()
    # print("二次编码YUV序列时间消耗%.3f秒" % (t2 - t1))
    return 0


def Extract_data_from_feature_txt(txtfile, frame_num):
    '''
    从txt文件中提取特征，该函数会被执行两次
    :param txtfile: 文件名称
    :return: 提取文件的数据，
    '''
    # print("正在读取特征文件\"{txtfile}\"...".format(txtfile=txtfile))
    t1 = time.time()
    list_pixel_residual = [[] for i in range(frame_num)]
    with open(txtfile, 'r', encoding='utf-8') as inf:
        lines = inf.readlines()[:-1]
        line_num = len(lines[4:-1])
        for i in range(line_num):
            line = lines[4 + i].split('\n')[0].strip().split(' ')
            i_frame = int(line[0])
            list_pixel_residual[i_frame] = list_pixel_residual[i_frame] + line[4:]
    t2 = time.time()
    # ("读取特征文件时间消耗%.3f秒" % (t2 - t1))
    # if "250" in txtfile:
    #     list_pixel_residual_250 = list_pixel_residual
    # elif "200" in txtfile:
    #     list_pixel_residual_200 = list_pixel_residual
    return list_pixel_residual


def Combine_feature(list_pixel_residual_250, list_pixel_residual_200):
    '''
    特征合并，以GOP=200特征为基础，使用GOP=250特征填补GOP=200特征
    :return:无
    '''
    t1 = time.time()
    for i in range(len(list_pixel_residual_200)):
        if not list_pixel_residual_200[i]:
            list_pixel_residual_200[i] = list(map(int, list_pixel_residual_250[i]))
    t2 = time.time()
    # print("特征处理时间消耗%.3f秒" % (t2 - t1))
    return list_pixel_residual_200


def Calculate_fluctuation_strength(list_pixel_residual, frame_width, frame_height):
    '''
    计算波动强度特征fluctuation_strength
    :return: 0
    '''
    # print("计算波动强度特征...")
    t1 = time.time()
    fluctuation_strength = []
    for i in range(len(list_pixel_residual)):
        if not list_pixel_residual[i]:
            fluctuation_strength.append(0)
        elif list_pixel_residual[i]:
            block_num_4x4 = int(len(list_pixel_residual[i]) / 16)
            block_4x4_std = []
            for j in range(block_num_4x4):
                block_4x4 = list_pixel_residual[i][16 * j:16 * (j + 1)]
                block_4x4 = np.array(list(map(int, block_4x4)))
                block_4x4_std.append(np.std(block_4x4, ddof=0))
            x = np.array(list(map(int, block_4x4_std)))
            block_4x4_num_in_frame = int((frame_height * frame_width) / 16)
            variance = ((x - np.mean(x)) ** 2).sum() / block_4x4_num_in_frame
            R = 1 - (1 / (1 + variance))
            fluctuation_strength.append(R)
    t2 = time.time()
    # print("波动强度特征生成时间消耗%.3f秒" % (t2 - t1))
    return fluctuation_strength


def Calculate_mean_motion_residual(list_pixel_residual, frame_width, frame_height):
    '''
    计算平均运动残差
    :return: 0
    '''
    # print("计算平均运动残差特征...")
    t1 = time.time()
    mean_motion_residual = []
    for i in range(len(list_pixel_residual)):
        if not list_pixel_residual[i]:
            mean_motion_residual.append(0)
        elif list_pixel_residual[i]:
            x = np.array(list(map(int, list_pixel_residual[i])))
            mean_motion_residual.append(x.sum() / (frame_width * frame_height))
    t2 = time.time()
    # print("平均运动残差特征生成时间消耗%.3f秒" % (t2 - t1))
    return mean_motion_residual


def show(video_out, fluctuation_strength, mean_motion_residual, frame_num, name):
    # x = [i for i in range(len(mean_motion_residual))]
    # y = data

    plt.subplot(1, 2, 1)
    plt.bar([i for i in range(frame_num)], mean_motion_residual)
    plt.title("mean motion residual")
    plt.xlabel("frame sequence")
    plt.subplot(1, 2, 2)
    plt.bar([i for i in range(frame_num)], fluctuation_strength)
    plt.title("residual fluctuation strength")
    plt.xlabel("frame sequence")
    # plt.subplot(1, 2, 3)
    # plt.bar([i for i in range(len(fluctuation_strength_window))], fluctuation_strength_window)
    # plt.title("fluctuation strength window")
    # plt.xlabel("frame sequence")
    # plt.subplot(2, 2, 4)
    # plt.bar([i for i in range(len(mean_motion_residual_window))], mean_motion_residual_window)
    # plt.title("mean motion residual window")
    # plt.xlabel("frame sequence")
    # plt.show()
    plt.savefig(os.path.join(video_out, name + '.jpg'))
    plt.close('all')


def Calculate_fluctuation_strength_window_mean(fluctuation_strength, frame_num):
    '''
    检测算法，计算移动窗口大小为5时的波动强度特征均值，得到第K帧的波动强度与移动窗口波动强度均值的比值
    :return:0
    '''
    # print("使用移动窗口算法计算波动强度特征均值，得到相对于相邻帧的波动强度均值异常点")
    t1 = time.time()
    window_range = 3
    fluctuation_strength_window = []
    fluctuation_strength_window.append(float(0))
    for k in range(1, len(fluctuation_strength)):
        R_mean = 1
        if k == 1:
            R_mean = (fluctuation_strength[3] + fluctuation_strength[4]) / 2
        elif k in range(1, window_range + 1) or k in range(frame_num - window_range, frame_num - 1):
            R_mean = (fluctuation_strength[k - 1] + fluctuation_strength[k + 1]) / 2
        elif k in range(window_range + 1, frame_num - window_range + 1):
            R_mean = sum(fluctuation_strength[k - window_range:k + window_range]) / (
                    2 * window_range)
        elif k == frame_num - 1:
            R_mean = (fluctuation_strength[k - 1] + fluctuation_strength[k - 2]) / 2
        if R_mean != 0:
            Y = abs(fluctuation_strength[k] / R_mean - 1)
        else:
            Y = 0
        if Y > 1.6:
            fluctuation_strength_window.append(k)

    t2 = time.time()
    # print("移动窗口算法计算异常值时间消耗%.3f秒" % (t2 - t1))
    return fluctuation_strength_window


def Calculate_mean_motion_residual_window_mean(mean_motion_residual, frame_num):
    mean_motion_residual_window = []
    mean_motion_residual_window.append(float(0))
    window_range = 3
    for k in range(1, len(mean_motion_residual)):
        R_mean = 1
        if k == 1:
            R_mean = (mean_motion_residual[3] + mean_motion_residual[4]) / 2
        elif k in range(1, window_range + 1) or k in range(frame_num - window_range, frame_num - 1):
            R_mean = (mean_motion_residual[k - 1] + mean_motion_residual[k + 1]) / 2
        elif k in range(window_range + 1, frame_num - window_range + 1):
            R_mean = sum(mean_motion_residual[k - window_range:k + window_range]) / (2 * window_range)
        elif k == frame_num - 1:
            R_mean = (mean_motion_residual[k - 1] + mean_motion_residual[k - 2]) / 2
        if R_mean != 0:
            Y = abs(mean_motion_residual[k] / R_mean - 1)
        else:
            Y = 0
        if Y > 0.4:
            mean_motion_residual_window.append(k)
    return mean_motion_residual_window


def detection(Input_file, temp_dir):
    # 第一步：去除编码器中的P帧的帧内预测代价计算部分，使得P帧只进行帧间编码，增强帧间特征
    # 第二步：待检测视频转化成YUV序列
    # 第三步：使用改进编码器对YUV序列进行首次编码得到feature_gop_250特征文件
    # 第四步：使用改进编码器对YUV序列进行二次编码得到feature_gop_200特征文件
    # 第五步：合并feature_gop_250特征和feature_gop_200特征得到除首帧之外所有帧的帧间编码P帧版本
    # 第六步：计算每一帧的平均运动残差和波动强度特征
    # 第七步：使用移动窗口算法计算每一帧的帧删除检测相对量化值
    # 第八步：阈值检测

    # 打印视频基本信息
    frame_num, frame_width, frame_height = Video_print_and_init(Input_file)
    # 将视频解压缩YUV格式
    name = Input_file.split('/')[-1].split('.')[0]
    YUV_file = temp_dir + '/' + name + ".yuv"
    Video_to_YUV(Input_file, YUV_file)
    # 使用新的编码器编码264文件，提取帧间运动残差特征
    H264_file1 = temp_dir + '/' + name + "_250.h264"
    H264_file2 = temp_dir + '/' + name + "_200.h264"
    Encoder_first_by_newx264(YUV_file, H264_file1, frame_width, frame_height)
    Encoder_second_by_newx264(YUV_file, H264_file2, frame_width, frame_height)
    # 提取特征文件，分块进行处理，并进行合并
    list_pixel_residual_250 = Extract_data_from_feature_txt("feature_gop_250.txt", frame_num)
    list_pixel_residual_200 = Extract_data_from_feature_txt("feature_gop_200.txt", frame_num)
    list_pixel_residual = Combine_feature(list_pixel_residual_250, list_pixel_residual_200)
    # 分别计算波动强度特征和平均运动残差
    fluctuation_strength = Calculate_fluctuation_strength(list_pixel_residual, frame_width, frame_height)
    mean_motion_residual = Calculate_mean_motion_residual(list_pixel_residual, frame_width, frame_height)
    # 移动窗口进行定位
    fluctuation_strength_window = Calculate_fluctuation_strength_window_mean(fluctuation_strength, frame_num)
    mean_motion_residual_window = Calculate_mean_motion_residual_window_mean(mean_motion_residual, frame_num)
    intersection = list(set(fluctuation_strength_window).intersection(set(mean_motion_residual_window)))
    show(temp_dir, fluctuation_strength, mean_motion_residual, frame_num, name)
    os.remove("feature_gop_250.txt")
    os.remove("feature_gop_200.txt")
    return intersection


if __name__ == '__main__':
    video = 'Z:/ljc/video_frame_dataset/frame_del_videos/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01_del.avi'
    video_out = 'Z:/ljc/video_frame_dataset/frame_del_detection/ApplyEyeMakeup'
    plt = detection(video, video_out)
    # plt_mean_motion_residual = plt[0]
    # plt_fluctuation_strength = plt[1]
    # plt_fluctuation_strength_window = plt[2]
    # plt_mean_motion_residual_window = plt[3]
    # # 检测结果图保存
    # plt_mean_motion_residual.savefig(os.path.join(video_out, 'motion_residual.jpg'))
