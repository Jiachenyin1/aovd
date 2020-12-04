from deep_sort.deep_sort import DeepSort  #在我们执行import时，当前目录是不会变的（就算是执行子目录的文件），还是需要完整的包名。


__all__ = ['DeepSort', 'build_tracker','build_tracker_car']  ##import * 的时候，防止导入过多的变量，把导入的限制在__all__中

def build_tracker(cfg, use_cuda):
    return DeepSort(cfg.DEEPSORT.REID_CKPT, 
                max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)
    

def build_tracker_car(cfg, use_cuda):
    return DeepSort(cfg.DEEPSORT.REID_CKPT_Car, 
                max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda,num_class = 685)







