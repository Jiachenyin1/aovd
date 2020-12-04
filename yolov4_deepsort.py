import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector,build_onnx
from deep_sort import build_tracker,build_tracker_car
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
# from threading import Thread
from dataset import LoadStreams
from detector.trt import tensorrt
import shutil

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        self.cuda_ctx  = None

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.datasets = LoadStreams(args.cam)
            self.cap = cv2.VideoCapture(args.cam)
        else:
            self.datasets = LoadStreams(args.VIDEO_PATH)
            self.cap = cv2.VideoCapture()   
        self.deepsort_person= build_tracker(cfg, use_cuda=use_cuda) 

    def __enter__(self):  #__enter__(self):当with开始运行的时候触发此方法的运行
        if isinstance(self.args.cam , int):
            if self.args.cam != -1:
                ret, frame = self.cap.read()
                assert ret, "Error: Camera error"
                self.im_width = frame.shape[0]
                self.im_height = frame.shape[1]
            else:
                assert os.path.isfile(self.video_path), "Path error"
                self.cap.open(self.video_path)
                self.im_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.im_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                assert self.cap.isOpened()
        elif isinstance(self.args.cam , str):
            self.cap.open(self.args.cam)
            self.im_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.cap.isOpened()


        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")
            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))
            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        count_P = []
        trt_person= tensorrt(self.cfg,[416,416])

        while self.cap.grab():
            ##socket
            for _, im0s, _ in self.datasets: ##dataset 进行了__next__ 方法
                start = time.time() 
                # _, ori_im = self.cap.retrieve()
                ori_im = im0s[0]
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                # bbox_xywh, cls_conf, cls_ids = self.session.forward(ori_im,self.im_width,self.im_height)
                ##这里非常重要, context  , buffer 全部应该在一个线程内实现,
                bbox_xywh, cls_conf, cls_ids = trt_person.detect(trt_person.context , trt_person.buffers,ori_im,self.im_width,self.im_height)
                # select person class  TODO
                class_det_P = [0]
                save_id = []
                for i, id in enumerate(cls_ids):
                    if id not in class_det_P:
                        save_id.append(i)
                ##numpy array 进行delete
                bbox_xywh_P = np.delete(bbox_xywh , [save_id],axis=0)
                cls_conf_P = np.delete(cls_conf ,[save_id])
                outputs_P ,count_num_P,detection_id_P= self.deepsort_person.update(bbox_xywh_P, cls_conf_P, im,count_P)

                # if len(outputs_P) > 0:
                ori_im,track_num,detection_id = draw_boxes(ori_im, outputs_P,count_num_P,detection_id_P,Type='person' )

                end = time.time()
                fps =  1 / (end - start)
                cv2.putText(ori_im, "FPS: %.2f" % (fps), (int(1050), int(200)), 0, 10e-3 * 200, (0, 255, 0), 2)

                if self.args.display:
                    cv2.imshow("test", ori_im)
                    wirte_img = cv2.resize(ori_im,(800,600))
                    cv2.waitKey(10)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str,default='MOT16-03.mp4')
    parser.add_argument("--config_detection", type=str, default="./configs/yolov4_trt.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true",default=True)
    parser.add_argument("--frame_interval", type=int, default=2)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int,default='-1')
    # parser.add_argument("--camera", action="store", dest="cam", type=str, default="rtsp://admin:abc12345@192.168.1.64/ch2/main/av_stream")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
