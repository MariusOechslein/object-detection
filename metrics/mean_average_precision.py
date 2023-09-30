import torch
from collections import Counter
from intersection_over_union import calc_iou

def mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold = 0.5,
        box_format = "corners",
        num_classes = 20
        ):
    '''
    Calculates the mean average precision for

    Note: This implementation is only for a single iou. For complete mAP algorithm, this function needs to be called for multiple ious.

    Parameters:
        pred_boxes (list): prediction of bboxes with [train_idx, class_pred, prob_score, x1, y1, x2, y2]
        true_boxes (list): real bboxes with [train_idx, class_pred, prob_score, x1, y1, x2, y2]
    '''

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # Create dictionary of the number of bboxes corresponding to each training_idx (which is the 0 index of a bbox)
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        # Convert entries of dictionary to torch.tensor containing zeros corresponding to the number of occurences. In order to keep track of covered bboxes.
        # Example result: amount_bboxes = { 0: torch.tensor([0,0,0]), 1: torch.tensor([0,0,0,0,0]) } for 3 bboxes for train_idx 0; and 5 bboxes for train_idx = 1
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Set up dataframes for iteration with logic
        detections.sort(key = lambda x: x[2], reverse = True) # Sort probability scores in descending order 
        true_positives = torch.zeros((len(detections))) # true positives := that the predicted bbox fits the ground_truth well/ the best.
        false_positives = torch.zeros((len(detections))) # false positives := that the predicted bbox doesn't fit the ground_truth well and should be disregarded and should be disregarded.
        total_true_bboxes = len(ground_truths)
        for detection_idx, detection in enumerate(detections):
            # Filter for all bboxes of one image (= training_idx)
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0.0 # iou = intersection_over_union

            # Go through all bboxes of this image and find the best intersection_over_union
            for idx, gt in enumerate(ground_truth_img):
                iou = calc_iou(
                        torch.tensor(detection[3:]), # Indeces for x1, y1, x2, y2
                        torch.tensor(gt[3:]),
                        box_format = box_format
                        )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0: # Logic for checking if it has been covered
                    # Since this predicted bbox is the best bbox for this ground_truth, keep track of it as a true positive 
                    true_positives[detection_idx] = 1 
                    # Signal that this ground_truth bboxes has been covered 
                    amount_bboxes[detection[0]][best_gt_idx] = 1 
                else:
                    # Since it is not the best bbox for this ground_truth index, keep track of it as false positive. 
                    false_positives[detection_idx] = 1 
            else:
                # Since iou threshold was not met, keep track of this predicted bbox as a false positive.
                false_positives[detection_idx] = 1

        # Calculate recall and precision by the cumulatives sums of false and true positives
        true_positives_cumsum = torch.cumsum(true_positives, dim=0)
        false_positives_cumsum = torch.cumsum(false_positives, dim=0)
        recalls = true_positives_cumsum / (total_true_bboxes + epsilon) # Recall = TP / (TP + FP)
        precisions = torch.divide(true_positives_cumsum, (true_positives_cumsum + false_positives_cumsum + epsilon)) # Precision = TP / (TP + FN)

        recalls = torch.cat((torch.tensor([0]), recalls)) # Needed for correct calculation, because we need to start at (0,0)
        precisions = torch.cat((torch.tensor([1]), precisions)) # Here we want to start at (0,1). Since recalls is x-axis and precisions is y-axis 

        average_precisions.append(torch.trapz(precisions, recalls)) # trapz() for area under the curve

    return sum(average_precisions) / len(average_precisions)




