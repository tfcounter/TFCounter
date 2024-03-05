import os
import torch
import torchvision
import argparse
import json
import numpy as np
import os
import copy
import time
import cv2
from tqdm import tqdm
from os.path import exists,join
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from utils import *
import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt




parser = argparse.ArgumentParser(description="TFCounter")
parser.add_argument("-dp", "--data_path", type=str, default='./dataset/BIKE_1000/', help="Path to the BIKE_1000 dataset")
parser.add_argument("-o", "--output_dir", type=str,default="./logsSave/BIKE_1000", help="/Path/to/output/logs/")
parser.add_argument("-ts", "--test-split", type=str, default='test', choices=["train", "test", "val"], help="what data split to evaluate on on")
parser.add_argument("-pt", "--prompt-type", type=str, default='box', choices=["box", "point", "text"], help="what type of information to prompt")
parser.add_argument("-d", "--device", type=str,default='cpu', help="device")
args = parser.parse_args()

 

if __name__=="__main__": 
    data_path = args.data_path
    anno_file = data_path + 'BIKE.json'
    data_split_file = data_path + 'Train_Test_Val_BIKE.json'
    im_dir = data_path + 'images'

    if not exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(args.output_dir+'/logs')
    
    if not exists(args.output_dir+'/%s'%args.test_split):
        os.mkdir(args.output_dir+'/%s'%args.test_split)

    if not exists(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type)):
        os.mkdir(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type))
    
    log_file = open(args.output_dir+'/logs/log-%s-%s.txt'%(args.test_split,args.prompt_type), "w")    

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)
    

    sam_checkpoint = "./pretrain/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    # sam_checkpoint = "./pretrain/sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(model=sam, fusion_type='mean', fusion_ratio=0.7)

    MAE = 0
    MAPE = 0
    RMSE = 0
    NAE = 0
    SRE = 0
    im_ids = data_split[args.test_split]
    for i,im_id in tqdm(enumerate(im_ids)):

        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        image = cv2.imread('{}/{}'.format(im_dir, im_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_prompt = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[1][0]
            y2 = bbox[1][1]
            if args.prompt_type=='box':
                # 确保x1, y1是左上角，x2, y2是右下角
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                input_prompt.append([x1, y1, x2, y2])
            elif args.prompt_type=='point':
                input_prompt.append([(x1+x2)//2, (y1+y2)//2])

        mask_list = []
        input_prompt_list = []
        le = 0
        while len(input_prompt)>0 and le<2:
            le+=1
            prompt_num = len(input_prompt)
            input_prompt_list += input_prompt

            masks = mask_generator.generate(image, input_prompt_list)

            filtered_masks = filter_masks(mask_list, masks, threshold=0.8) 
            filtered_masks = filter_overlapping_masks(filtered_masks, filtered_masks)
            filtered_masks = filter_overlapping_masks(filtered_masks, mask_list) 
            mask_list += filtered_masks
            mask_list,  input_prompt_list = filter_overlapping_masks_list(mask_list, input_prompt_list) 

            
            filtered_masks.sort(key=lambda x: x['predicted_iou'], reverse=True)
            for mask in filtered_masks[:min(2, len(filtered_masks))]:
                true_indices = np.where(mask['segmentation'])
                box = [true_indices[1].min(), true_indices[0].min(), true_indices[1].max(), true_indices[0].max()]
                if not is_box_overlapping(box, input_prompt_list, iou_threshold=0.5):
                    input_prompt.append(box)

            input_prompt = input_prompt[prompt_num:]
            
 

        gt_cnt = dots.shape[0]
        pred_cnt = len(mask_list)

        print(pred_cnt, gt_cnt, abs(pred_cnt-gt_cnt))
        log_file.write("%d: %d,%d,%d\n"%(i, pred_cnt, gt_cnt,abs(pred_cnt-gt_cnt)))
        log_file.flush()

        err = abs(gt_cnt - pred_cnt)
        MAE = MAE + err
        RMSE = RMSE + err**2
        NAE = NAE+err/gt_cnt  # MAPE = MAPE + err*100/gt_cnt
        SRE = SRE+err**2/gt_cnt

        """Mask visualization
        plt.figure(figsize=(10,10))
        image = cv2.imread('{}/{}'.format(im_dir, im_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(image)
        show_anns(mask_list, plt.gca())
        # for input_box in input_prompt_list:
        #     show_box(input_box, plt.gca())
        plt.savefig('%s/%s/%03d_mask.png'%(args.output_dir,args.test_split,i), bbox_inches='tight', pad_inches=0)
        plt.close()"""
        
    
    MAE = MAE/len(im_ids)
    RMSE = math.sqrt(RMSE/len(im_ids))
    NAE = NAE/len(im_ids)
    SRE = math.sqrt(SRE/len(im_ids))

    print(len(im_ids))
    print("MAE:%0.2f,RMSE:%0.2f,MAPE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.write("MAE:%0.2f,RMSE:%0.2f,NAE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.close()



    

    

    

        
