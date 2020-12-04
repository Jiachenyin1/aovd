import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, output=None, count = [] , detection_id=0,Type='car',offset=(0,0)  ):
    track_num = len(set(count))
    if len(output) != 0:
        bbox =  output[:, :4]
        identities =  output[:, -1]
        detection_id = len(identities)
        for i,box in enumerate(bbox):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0    
            color = compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
            cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)  ##填充
            cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        ##  将track 跟踪数量放到图上
    else:
        detection_id = 0
    puttxt_height=img.shape[0]
    puttxt_width=img.shape[1]
    if Type=='car':
        cv2.putText(img, "Total Car: " + str(track_num), (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(img, "Current Car Counter: " + str(detection_id), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
    else:
        cv2.putText(img, "Total Person: " + str(track_num), (int(4), int(25)), 0, 1, (255, 0, 255), 2)
        cv2.putText(img, "Current Person Counter: " + str(detection_id), (int(4), int(50)), 0, 1, (255, 0, 255), 2)
    # cv2.putText(img, "FPS: %.2f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
    return img,track_num,detection_id






if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
