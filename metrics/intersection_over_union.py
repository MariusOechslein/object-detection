import torch


def calc_iou(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union of two bounding boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4).
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4).
        box_format (str): how bbox are described. midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2).

    Returns:
        tensor: Intersection over union for all examples.
    """

    assert torch.is_tensor(boxes_preds) and torch.is_tensor(boxes_labels) and isinstance(box_format, str)

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # '...' indexing to keep additional dimensions
    # Yolo algorithm dimesions: (N, S, S, 4)
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2 
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        raise ValueError("box_format has to be 'midpoint' or 'corners'")

    # Calculate the corner points of intersection of the bounding boxes
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) # .clamp(0) to handle the case of zero intersection. Would falsely return positive intersection otherwise.
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area

    return intersection / (union - intersection + 1e-6) # +1e-6 to prevent accidentally dividing by zero because of rounding error.  
