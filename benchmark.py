import os
import numpy as np
import cv2

from tqdm import tqdm

def calculate_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union != 0 else 0

def process_folder(predictions_path, ground_truth_path):
    """
    Compute the average IoU for all masks in the dataset.
    """
    categories = os.listdir(predictions_path)
    iou_scores = []
    
    for category in categories:
        pred_category_path = os.path.join(predictions_path, category)
        gt_category_path = os.path.join(ground_truth_path, category)
        
        if not os.path.isdir(pred_category_path) or not os.path.isdir(gt_category_path):
            continue
        
        for file in tqdm(os.listdir(pred_category_path)):
            if file.endswith("_mask.jpg"):
                pred_mask_path = os.path.join(pred_category_path, file)
                gt_mask_path = os.path.join(gt_category_path, file)
                
                if not os.path.exists(gt_mask_path):
                    print(f"Warning: Ground truth mask {gt_mask_path} not found.")
                    continue
                
                # Load masks as binary images
                pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE) > 40
                gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE) > 40
                
                # Resize predicted mask to match ground truth size if necessary
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                
                iou = calculate_iou(pred_mask, gt_mask)
                iou_scores.append(iou)
    
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    print(f"Average IoU: {avg_iou:.4f}")
    return avg_iou

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate IoU for predicted and ground truth masks.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions dataset.")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to ground truth dataset.")
    
    args = parser.parse_args()
    
    process_folder(args.predictions, args.ground_truth)
