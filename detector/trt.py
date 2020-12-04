import sys
import os
import time
import argparse
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from detector.YOLOv3.nms import boxes_nms
import torch

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

# __all__ = ['trt']

class tensorrt():
    def __init__(self , cfg, img_size = [416,416]):
        self.cfg = cfg
        self.engine =  self.get_engine(cfg.YOLOV4.WEIGHT)
        self.context =  self.engine.create_execution_context()
        self.buffers = self.allocate_buffers(self.engine, 1)
        IN_IMAGE_H, IN_IMAGE_W = img_size
        self.context.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))
        self.num_classes = 80
        self.image_size = img_size
        # return context , buffers

    def get_engine(self,engine_path):
    # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_path))
        TRT_LOGGER = trt.Logger()
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def detect(self,context , buffers , image_src,video_width=416,video_height=416):
        IN_IMAGE_H, IN_IMAGE_W = self.image_size
        ta = time.time()
    # Input
        # image_src = cv2.imread(image_src)
        resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        img_in = np.ascontiguousarray(img_in)
        print("Shape of the network input: ", img_in.shape)
        # print(img_in)

        inputs, outputs, bindings, stream = buffers
        # print('Length of inputs: ', len(inputs))
        inputs[0].host = img_in

        trt_outputs = self.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        # print('Len of outputs: ', len(trt_outputs))

        trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
        trt_outputs[1] = trt_outputs[1].reshape(1, -1, self.num_classes)

        tb = time.time()

        # print('-----------------------------------')
        # print('    TRT inference time: %f' % (tb - ta))
        # print('-----------------------------------')

        # boxes = post_processing(img_in, 0.4, 0.6, trt_outputs)
        self.boxes = np.array(self.post_processing(img_in, self.cfg.YOLOV4.SCORE_THRESH, self.cfg.YOLOV4.NMS_THRESH, trt_outputs))[0]
        # assert self.boxes[:,0:4]
        self.box =self.xyxy_to_xywh(self.boxes[:,0:4])
        self.box = self.box * np.array([video_width, video_height, video_width, video_height])

        self.cls = self.boxes[:,5]
        self.id = self.boxes[:,6]
        return self.box , self.cls  , self.id

    def post_processing(self,img, conf_thresh, nms_thresh, output):

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

        # print('-----------------------------------')
        # print('       max and argmax : %f' % (t2 - t1))
        # print('                  nms : %f' % (t3 - t2))
        # print('Post processing total : %f' % (t3 - t1))
        # print('-----------------------------------')
    
        return bboxes_batch

        # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    # Simple helper data class that's a little nicer to use than a 2-tuple.
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    def allocate_buffers(self,engine, batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:

            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dims = engine.get_binding_shape(binding)
        
        # in case batch dimension is -1 (dynamic)
            if dims[0] < 0:
                size *= -1
        
            dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
        # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(self,context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
        stream.synchronize()
    # Return only the host outputs.
        return [out.host for out in outputs]
    def GiB(self,val):
        return val * 1 << 30

    def xyxy_to_xywh(self,boxes_xyxy):
        if isinstance(boxes_xyxy, torch.Tensor):
            boxes_xywh = boxes_xyxy.clone()
        elif isinstance(boxes_xyxy, np.ndarray):
            boxes_xywh = boxes_xyxy.copy()

        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        return boxes_xywh

