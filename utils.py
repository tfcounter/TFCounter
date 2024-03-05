import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import math
from torchvision import transforms
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('agg')
import scipy.spatial as T
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label, find_contours
from selective_search import selective_search
from segment_anything import sam_model_registry, SamPredictor

preprocess =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512), interpolation='BICUBIC'), 
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_anns(anns,ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        #print(m.shape)
        #print(np.unique(m))
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]#np.array([30/255, 144/255, 255/255])#
        # color_mask = np.array([1, 1, 1])
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.8)))
        # ax.imshow(np.dstack((img, m*0.5)))

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

def get_boxes_from_sim(similarity_mask):
    _, similarity_mask = cv2.threshold(similarity_mask, np.max(similarity_mask)/1.1, 1, cv2.THRESH_BINARY)
    contours = find_contours(similarity_mask)
    boxes = []
    scores = []
    max_score = 0
    best_box = None
    for contour in contours:
        Xmin = int(np.min(contour[:,1]))
        Xmax = int(np.max(contour[:,1]))
        Ymin = int(np.min(contour[:,0]))
        Ymax = int(np.max(contour[:,0]))

        score = np.sum(similarity_mask[Ymin:Ymax,Xmin:Xmax])/((Xmax-Xmin)*(Ymax-Ymin))
        if score>max_score:
            max_score = score
            best_box = [Xmin,Ymin,Xmax,Ymax]

        if score>0.5 and (Xmax-Xmin>=5) and (Ymax-Ymin>=5):
            boxes.append([Xmin,Ymin,Xmax,Ymax])
            scores.append(score)
    
    if len(boxes)>1:
        boxes, _ = nms(boxes, scores, threshold=0.01)
    elif len(boxes)<1:
        boxes.append(best_box)
    
    boxes_new = []
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
    
        x = (x1+x2)//2
        y = (y1+y2)//2

        h = x2-x1
        w = y2-y1

        if h<16:
            h=16
            x1 = x-h//2
            x2 = x+h//2
        if w<16:
            w=16
            y1 = y-w//2
            y2 = y+w//2

        boxes_new.append([x1,y1,x2,y2])
    
    return boxes_new

def select_max_region(img_gray):
    #find contour
    labeled_img, num = label(img_gray, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        sub_num = np.sum(labeled_img==i)
        if sub_num > max_num:
            max_num = sub_num
            max_label = i

    if max_label > 0:
        img_gray[labeled_img!=max_label] = 0
    
    contour = find_contours(img_gray)[0]
    pnum = contour.shape[0]
    Xmin = np.min(contour[:,1])
    Xmax = np.max(contour[:,1])
    Ymin = np.min(contour[:,0])
    Ymax = np.max(contour[:,0])

    h = Xmax-Xmin
    w = Ymax-Ymin

    boxes = []
    k_ = max(h,w)//32
    if k_<2:
        boxes.append([Xmin,Ymin,Xmax,Ymax])
        return img_gray, boxes
    else:
        cnum = int(pnum//k_)
        scores = []
        max_score = 0
        best_box = None
        contour_h = np.sort(contour,axis=0)
        for i in range(int(pnum//cnum)):
            Xmin = int(np.min(contour_h[i*cnum:(i+1)*cnum,1]))
            Xmax = int(np.max(contour_h[i*cnum:(i+1)*cnum,1]))
            Ymin = int(np.min(contour_h[i*cnum:(i+1)*cnum,0]))
            Ymax = int(np.max(contour_h[i*cnum:(i+1)*cnum,0]))

            score = np.sum(img_gray[Ymin:Ymax,Xmin:Xmax])/((Xmax-Xmin)*(Ymax-Ymin))
            if score>max_score:
                max_score = score
                best_box = [Xmin,Ymin,Xmax,Ymax]
            if score>=0.3 and (Xmax-Xmin>=5) and (Ymax-Ymin>=5):
                boxes.append([Xmin,Ymin,Xmax,Ymax])
                scores.append(score)
        
        contour_v = np.sort(contour,axis=1)
        for i in range(int(pnum//cnum)):
            Xmin = int(np.min(contour_v[i*cnum:(i+1)*cnum,1]))
            Xmax = int(np.max(contour_v[i*cnum:(i+1)*cnum,1]))
            Ymin = int(np.min(contour_v[i*cnum:(i+1)*cnum,0]))
            Ymax = int(np.max(contour_v[i*cnum:(i+1)*cnum,0]))

            score = np.sum(img_gray[Ymin:Ymax,Xmin:Xmax])/((Xmax-Xmin)*(Ymax-Ymin))
            if score>max_score:
                max_score = score
                best_box = [Xmin,Ymin,Xmax,Ymax]
            if score>=0.3 and (Xmax-Xmin>=5) and (Ymax-Ymin>=5):
                boxes.append([Xmin,Ymin,Xmax,Ymax])
                scores.append(score)
        
        if len(boxes)>1:
            boxes, _ = nms(boxes, scores, threshold=0.01)
        elif len(boxes)<1:
            boxes.append(best_box)
        
    boxes_new = []
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
    
        x = (x1+x2)//2
        y = (y1+y2)//2

        h = x2-x1
        w = y2-y1

        if h<16:
            h=16
            x1 = x-h//2
            x2 = x+h//2
        if w<16:
            w=16
            y1 = y-w//2
            y2 = y+w//2

        boxes_new.append([x1,y1,x2,y2])
                    
    return img_gray, boxes_new



def generate_ref_info(args):
    im_dir = args.data_path + 'Images'
    exam_dir = args.data_path + 'exemplar.txt'

    sam_checkpoint = "./pretrain/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    with open(exam_dir) as f:
        exa_ids = f.readlines()
    
    target_feats=[]
    target_embeddings = []
    mask_sizes = []
    for i,exa_id in enumerate(exa_ids):
        strings = exa_id.strip().split(':')
        im_id = strings[0]
        ref_bbox = strings[1][1:-1].split(', ')
        ref_bbox = [int(box) for box in ref_bbox]

        image = cv2.imread('{}/{}'.format(im_dir, im_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_size = image.shape[:2]

        predictor.set_image(image)

        if args.prompt_type=='box':
            ref_bbox = np.array([ref_bbox[1],ref_bbox[0],ref_bbox[3],ref_bbox[2]])
            ref_bbox = torch.tensor(ref_bbox, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(ref_bbox, img_size)
            masks, iou_preds, low_res_masks = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
                )
        elif args.prompt_type=='point':
            ref_points = np.array([[(ref_bbox[1]+ref_bbox[3])//2,(ref_bbox[0]+ref_bbox[2])//2]])
            ref_points = torch.tensor(ref_points, device=predictor.device)
            transformed_points = predictor.transform.apply_coords_torch(ref_points, img_size)
            in_labels = torch.ones(transformed_points.shape[0], dtype=torch.int, device=predictor.device)
            masks, iou_preds, low_res_masks = predictor.predict_torch(
                point_coords=transformed_points[:,None,:],
                point_labels=in_labels[:,None],
                boxes=None,
                multimask_output=False
                )

        mask_size = math.sqrt(np.sum(masks[0].cpu().float().numpy()))
        mask_sizes.append(mask_size)

        feat = predictor.get_image_embedding().squeeze()
        ref_feat = feat.permute(1, 2, 0)

        low_res_masks = F.interpolate(low_res_masks, size=ref_feat.shape[0: 2], mode='bilinear', align_corners=False)
        low_res_masks = low_res_masks.flatten(2, 3)
        masks_low_res = (low_res_masks > predictor.model.mask_threshold).float()
        topk_idx = torch.topk(low_res_masks, 1)[1]
        masks_low_res.scatter_(2, topk_idx, 1.0)
        ref_mask = masks_low_res[0].cpu()
        ref_mask = ref_mask.squeeze().reshape(ref_feat.shape[0: 2])

        # Target feature extraction
        target_feat = ref_feat[ref_mask > 0]
        if target_feat.shape[0]>0:
            target_embedding = target_feat.mean(0).unsqueeze(0)
            target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
            target_embedding = target_embedding.unsqueeze(0)
            target_feats.append(target_feat)
            target_embeddings.append(target_embedding)

    mask_size = np.array(mask_sizes).min(0)
    target_feat = torch.mean(torch.cat(target_feats, dim=0), dim=0, keepdim=True)
    target_embedding = torch.mean(torch.cat(target_embeddings, dim=0), dim=0, keepdim=True)

    return target_feat, target_embedding, mask_size


def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou
    
def filter_masks(mask_list1, mask_list2, threshold=0.9):
    if mask_list1==[]: 
        return mask_list2
    filtered_mask_list2 = []
    for mask2 in mask_list2:
        iou_list = [calculate_iou(mask1['segmentation'], mask2['segmentation']) for mask1 in mask_list1]
        if max(iou_list) < threshold:
            filtered_mask_list2.append(mask2)
    return filtered_mask_list2

def filter_overlapping_masks(mask_list1, mask_list2):
    filtered_mask_list = []
    for mask1 in mask_list1:
        is_covered = False
        for mask2 in mask_list2:
            if mask1['area'] < mask2['area'] and np.all(np.logical_and(mask1['segmentation'], mask2['segmentation']) == mask1['segmentation']):
                is_covered = True
                break
        if not is_covered:
            filtered_mask_list.append(mask1)
    return filtered_mask_list

def filter_overlapping_masks_list(mask_list, input_prompt_list):
    filtered_mask_list = []
    for mask1 in mask_list:
        is_covered = False
        for mask2 in mask_list:
            if mask1['area'] < mask2['area'] and np.all(np.logical_and(mask1['segmentation'], mask2['segmentation']) == mask1['segmentation']):
                is_covered = True
                true_indices = np.where(mask1['segmentation'])
                box = [true_indices[1].min(), true_indices[0].min(), true_indices[1].max(), true_indices[0].max()]
                while box in input_prompt_list:
                    input_prompt_list.remove(box)
                break
        if not is_covered:
            filtered_mask_list.append(mask1)
    return filtered_mask_list, input_prompt_list

def box_iou(box1, box2):
    # 计算矩形框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算矩形框的交集面积
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算交并比
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

def is_box_overlapping(box, box_list, iou_threshold=0.8):
    for other_box in box_list:
        iou = box_iou(box, other_box)
        if iou > iou_threshold:
            return True
    return False
