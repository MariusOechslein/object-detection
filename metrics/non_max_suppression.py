import torch

from intersection_over_union import calc_iou

def nms(
        bboxes, 
        threshold,
        iou_threshold, 
        box_format="corners"
    ):
    '''
    Sorts out the most likely bounding box of multiple proposed bounding boxes.

    Parameters:
        predictions (list): list of multiple prediction: [class_label, probability, x1, y1, x2, y2]. 
        threshold (float): probability threshold for which bboxes should be removed up front.
        iou_threshold (float): threshold for how much bboxes of one class are allowed to overlap, before they are removed in favor for the more probable bbox.
        box_format (str): 

    Returns:
        List of most likely bounding boxes.
    '''

    assert isinstance(bboxes, list)

    # Only keep boxes which have a probability of > threshold
    bboxes = [box for box in bboxes if box[1] > threshold]

    bboxes_after_nms = []
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # Sort bboxes by probability descending

    while bboxes:
        chosen_box = bboxes.pop(0) # Choose most likely bbox of the remaining bboxes

        # Remove bboxes of the same class with less probablity and iou of > iou_threshold. Note: Watch out to not compare bboxes of different classes! 
        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or calc_iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format = box_format) < iou_threshold]

        # Append the chosen_box to the final list, since this is one of the most likely bboxes that we want to keep.
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

