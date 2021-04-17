# -*- coding: utf-8 -*-
# @Author  : Abner
# @Time    : 2018/12/19

import os
from scipy import misc as scisc
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse
import imageio

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='D:\\CASIA-B\\', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='D:\\Gait\\', type=str,
                    help='Root path for output.')
parser.add_argument('--log_file', default='./pretreatment.log', type=str,
                    help='Log file path. Default: ./pretreatment.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
parser.add_argument('--worker_num', default=1, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 1')
opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num

T_H = 64
T_W = 64


def log2str(pid, comment, logs):#comment:START/FINISH/FAIL/WARNING 函数功能:返回输出文件的创建是否成功
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    return str_log


def log_print(pid, comment, logs):#comment:START/FINISH/FAIL/WARNING  logs:文件路径
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:#以追加模式打开文件
            log_f.write(str_log)
    if comment in [START, FINISH]:#每执行500个进程输出一次进程个数
        if pid % 500 != 0:
            return
    print(str_log, end='')#输出进程执行情况


def cut_img(img, seq_info, frame_name, pid):#img:图片数据 seq_info:步态序列所在文件夹 frame_name:图片所在路径 pid:第几个进程
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    if img.sum() <= 10000:
        message = 'seq:%s, frame:%s, no data, %d.' % (
            '-'.join(seq_info), frame_name, img.sum())
        warn(message)
        log_print(pid, WARNING, message)#图片数据错误
        return None
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        message = 'seq:%s, frame:%s, no center.' % (
            '-'.join(seq_info), frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')


def cut_pickle(seq_info, pid):#seq_info数据集的路径
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)#START:START
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    frame_list = os.listdir(seq_path)
    frame_list.sort()
    count_frame = 0
    for _frame_name in frame_list:#frame_list:文件中每张图片的路径
        frame_path = os.path.join(seq_path, _frame_name)
        img = cv2.imread(frame_path)[:, :, 0]#读取图片
        img = cut_img(img, seq_info, _frame_name, pid)
        if img is not None:#如果图片数据没有错误，保存图片处理结果
            # Save the cut img
            save_path = os.path.join(out_dir, _frame_name)
            imageio.imwrite(save_path, img)#scisc.imsave(save_path, img)
            count_frame += 1
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))

if __name__ == '__main__':
    pool = Pool(WORKERS)
    results = list()
    pid = 0

    print('Pretreatment Start.\n'
          'Input path: %s\n'
          'Output path: %s\n'
          'Log file: %s\n'
          'Worker num: %d' % (
              INPUT_PATH, OUTPUT_PATH, LOG_PATH, WORKERS))

    id_list = os.listdir(INPUT_PATH)#返回指定的文件夹包含的文件或文件夹的名字的列表
    id_list.sort()
    # Walk the input path
    for _id in id_list:
        seq_type = os.listdir(os.path.join(INPUT_PATH, _id))#D:\INUPUT_PATH:GaitDatasetB-silh  seq_type:D:\GaitDatasetB-silh\001
        seq_type.sort()
        for _seq_type in seq_type:
            view = os.listdir(os.path.join(INPUT_PATH, _id, _seq_type))#view:D:\GaitDatasetB-silh\001\bg-01
            view.sort()
            for _view in view:
                seq_info = [_id, _seq_type, _view]#seq-info:D:\GaitDatasetB-silh\001\bg-01\000
                out_dir = os.path.join(OUTPUT_PATH, *seq_info)#OUTPUT_PATH:D:\Gait\
                os.makedirs(out_dir)#创建目录
                results.append(
                    pool.apply_async(
                        cut_pickle,
                        args=(seq_info, pid)))
                sleep(0.02)
                pid += 1#正在进行的任务是第几项任务

    pool.close()
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get(timeout=0.1)
            except Exception as e:
                if type(e) == MP_TimeoutError:
                    unfinish += 1
                    continue
                else:
                    print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                          i, type(e))
                    raise e
    pool.join()
