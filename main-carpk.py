import os
import torch
import torchvision
import argparse
import json
import numpy as np
import os
import copy
from tqdm import tqdm
from os.path import exists,join
import pickle
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.automatic_mask_generator_carpk import SamAutomaticMaskGenerator as SamAutomaticMaskGenerator_carpk
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from utils import * 

parser = argparse.ArgumentParser(description="TFCounter")
parser.add_argument("-dp", "--data_path", type=str, default='./dataset/CARPK/CARPK_devkit/data/', help="Path to the CARPK dataset")
parser.add_argument("-o", "--output_dir", type=str,default="./logsSave/CARPK", help="/Path/to/output/logs/")
parser.add_argument("-ts", "--test-split", type=str, default='test', choices=["train", "test"], help="what data split to evaluate on on")
parser.add_argument("-pt", "--prompt-type", type=str, default='box', choices=["box", "point"], help="what type of information to prompt")
parser.add_argument("-d", "--device", type=str,default='cpu', help="device")
args = parser.parse_args()
        
if __name__=="__main__": 
    data_path = args.data_path
    anno_file = data_path + 'Annotations'
    data_split_file = data_path + 'ImageSets'
    im_dir = data_path + 'Images'

    if not exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(args.output_dir+'/logs')
    
    if not exists(args.output_dir+'/%s'%args.test_split):
        os.mkdir(args.output_dir+'/%s'%args.test_split)

    if not exists(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type)):
        os.mkdir(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type))
    
    log_file = open(args.output_dir+'/logs/log-%s-%s.txt'%(args.test_split,args.prompt_type), "w") 

    sam_checkpoint = "./pretrain/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    
    mask_generator_carpk = SamAutomaticMaskGenerator_carpk(model=sam)
    mask_generator = SamAutomaticMaskGenerator(model=sam, fusion_type='max', fusion_ratio=0.7)
    
    ref_info = generate_ref_info(args)

    MAE = 0
    RMSE = 0
    NAE = 0
    SRE = 0
    with open(data_split_file+'/%s.txt'%args.test_split) as f:
        im_ids = f.readlines()
    
    for i,im_id in tqdm(enumerate(im_ids)):
        im_id = im_id.strip()
        image = cv2.imread('{}/{}.png'.format(im_dir, im_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        mask_list = []
        input_prompt_list = []
        input_prompt = []
        le = 0
        while le==0 or (len(input_prompt)>0 and le<2):
            if le==0:
                masks = mask_generator_carpk.generate(image, ref_info)
            else:
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

            input_prompt = input_prompt[prompt_num:] if le>0 else input_prompt
            le+=1
        
        
        with open(anno_file+'/%s.txt'%im_id) as f:
                    box_lines = f.readlines()
        gt_cnt = len(box_lines)
        pred_cnt = len(mask_list)

        print(pred_cnt, gt_cnt, abs(pred_cnt-gt_cnt))
        log_file.write("%d: %d,%d,%d\n"%(i, pred_cnt, gt_cnt,abs(pred_cnt-gt_cnt)))
        log_file.flush()

        err = abs(gt_cnt - pred_cnt)
        MAE = MAE + err
        RMSE = RMSE + err**2
        NAE = NAE+err/gt_cnt
        SRE = SRE+err**2/gt_cnt

        """
        plt.figure(figsize=(10,10))
        plt.axis('off')
        ax = plt.gca()
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        plt.imshow(image)
        show_anns(mask_list, plt.gca())
        plt.savefig('%s/%s/%03d_mask.png'%(args.output_dir,args.test_split,i), bbox_inches='tight', pad_inches=0)
        plt.close()#"""

    MAE = MAE/len(im_ids)
    RMSE = math.sqrt(RMSE/len(im_ids))
    NAE = NAE/len(im_ids)
    SRE = math.sqrt(SRE/len(im_ids))

    print("MAE:%0.2f,RMSE:%0.2f,NAE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.write("MAE:%0.2f,RMSE:%0.2f,NAE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.close()

        
