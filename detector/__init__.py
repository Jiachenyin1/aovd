from .YOLOv3 import YOLOv3
import onnxruntime
import numpy as np 
import time
import cv2
import torch
from detector.YOLOv3.nms import boxes_nms
__all__ = ['build_detector','build_onnx']

def xyxy_to_xywh(boxes_xyxy):
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xywh = boxes_xyxy.clone()
    elif isinstance(boxes_xyxy, np.ndarray):
        boxes_xywh = boxes_xyxy.copy()

    boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
    boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    return boxes_xywh

def build_detector(cfg, use_cuda):
    return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, 
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                    is_xywh=True, use_cuda=use_cuda)


class build_onnx():
    def __init__(self , cfg):

        self.session = onnxruntime.InferenceSession(cfg.YOLOV4.WEIGHT)
        # session = onnx.load(onnx_path)
        print("The model expects input shape: ", self.session.get_inputs()[0].shape)
        self.class_names = self.load_class_names(cfg.YOLOV4.CLASS_NAMES)


    def forward(self,img,video_width,video_height):
        IN_IMAGE_H =  self.session.get_inputs()[0].shape[2]
        IN_IMAGE_W =  self.session.get_inputs()[0].shape[3]

    # Input
        resized = cv2.resize(img, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        print("Shape of the network input: ", img_in.shape)

        # Compute
        input_name = self.session.get_inputs()[0].name
        t5 = time.time()
        outputs = self.session.run(None, {input_name: img_in})
        t6 = time.time()
        print('       -------------infer----------------: %f' % (t5 - t6))


        self.boxes = np.array(self.post_processing(img_in, 0.4, 0.6, outputs))[0]
        self.box =xyxy_to_xywh(self.boxes[:,0:4])
        self.box = self.box * np.array([video_width, video_height, video_width, video_height])

        self.cls = self.boxes[:,5]
        self.id = self.boxes[:,6]
        return self.box , self.cls  , self.id

    def load_class_names(self , namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names

    def post_processing(self,img, conf_thresh, nms_thresh, output):

    # [batch, num, 1, 4]
        box_array = output[0]
    # [batch, num, num_classes]
        confs = output[1]

        t1 = time.time()

        if type(box_array).__name__ != 'ndarray':
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        num_classes = confs.shape[2]

    # [batch, num, 4]
        box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        t2 = time.time()

        bboxes_batch = []
        for i in range(box_array.shape[0]):
       
            argwhere = max_conf[i] > conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            bboxes = []
        # nms for each class
            for j in range(num_classes):

                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                keep = np.array(boxes_nms(torch.tensor(ll_box_array), torch.tensor(ll_max_conf), nms_thresh))
            
                if (keep.size > 0):
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
        
            bboxes_batch.append(bboxes)

        t3 = time.time()

        print('-----------------------------------')
        print('       max and argmax : %f' % (t2 - t1))
        print('                  nms : %f' % (t3 - t2))
        print('Post processing total : %f' % (t3 - t1))
        print('-----------------------------------')
    
        return bboxes_batch
