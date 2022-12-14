from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import cv2
import os
import numpy as np

from modules.models import RetinaFaceModel
from modules.utils import (load_yaml, pad_input_image, recover_pad_output)

#flags.DEFINE_string('cfg_path', './configs/retinaface_mbv2.yaml',
#                    'config file path')
#flags.DEFINE_string('img_path', '', 'path to input image')
#flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
#flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
#flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')

def draw_bbox(img, ann, img_height, img_width):
    """draw bboxes and landmarks"""
    # bbox
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # confidence
    text = "{:.4f}".format(ann[15])
    cv2.putText(img, text, (int(ann[0] * img_width), int(ann[1] * img_height)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

def main(img_input, iou_th=0.4,score_th=0.5,down_scale_factor=1.0):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg = load_yaml("./anime-face-detector-main/configs/retinaface_mbv2.yaml")

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=iou_th,
                            score_th=score_th)

    # load checkpoint
    checkpoint_dir = './anime-face-detector-main/checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    #if not os.path.exists(img_path):
    #    print(f"cannot find image path from {img_path}")
    #    exit()

    #print("[*] Processing on single image {}".format(img_path))

    img_raw = img_input
    img_height_raw, img_width_raw, _ = img_raw.shape
    img = np.float32(img_raw.copy())

    if down_scale_factor < 1.0:
        img = cv2.resize(img, (0, 0), fx=down_scale_factor,
                            fy=down_scale_factor,
                            interpolation=cv2.INTER_LINEAR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))  

    # run model
    outputs = model(img[np.newaxis, ...]).numpy()

    # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)

    res = []
    for prior_index in range(len(outputs)):
        ann = outputs[prior_index]
        x1, y1, x2, y2 = int(ann[0] * img_width_raw), int(ann[1] * img_height_raw), \
                         int(ann[2] * img_width_raw), int(ann[3] * img_height_raw)
        
        x, y, w, h = x1, y1, x2-x1, y2-y1
        if(h>w):
            y = y-round((h-w)/2)
            w = h
        elif(w>h):
            x = x-round((w-h)/2)
            h = w
        
        #?????????????????????
        k=1.5
        x = x-round(w*(k-1)/2)
        y = y-round(h*(k-1)/2)
        w = round(w*k)
        h = w

        temp_res = [x, y, w, h]
        res.append(temp_res)

    #????????????????????????????????????????????????????????????????????????
    res_nd = np.array(res)
    md = np.percentile(res_nd[:,2],50)
    sm = res_nd[:,2]>md/1.2
    res = res_nd[sm].tolist()

    return(res)

    # draw and save results
    #save_img_path = os.path.join('out_' + os.path.basename(FLAGS.img_path))
    #for prior_index in range(len(outputs)):
    #    draw_bbox(img_raw, outputs[prior_index], img_height_raw,
    #                    img_width_raw)
    #cv2.imwrite(save_img_path, img_raw)
    #print(f"[*] save result at {save_img_path}")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
