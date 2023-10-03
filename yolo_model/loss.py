import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_noobj = 0.5 # Constant from the paper 
        self.lambda_coord = 5 # Constant from the paper 
    
    def forward(self, predictions, target):
        '''
        Calculates the (multi-part) loss function of the YOLO model. 
        '''
        # Gettings labels into the correct format: (N, S, S, 25)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # Getting the best predicted bounding box and the real bounding box to be able to calculate the loss
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0) # Get best bbox by argmax of ious. This bbox should then be responsible for that cell.
        exists_box = target[..., 20].unsqueeze(3) # From paper: I * obj_i. Whether this is actually an object in the cell.

        ### Calculate the parts of the loss function
        # For box coordinates
        box_predictions = exists_box * (
            (
                best_box * predictions[..., 26:30]
                + (1 - best_box) * predictions[..., 21:25]
            ) 
        )
        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6)) # Be careful to prevent negative results and dividing by 0
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # For object loss
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box), # Converting to format which mse function expects. 
            torch.flatten(exists_box * target[..., 20:21]),
        ) 

        # For no object (with interpretation of paper that the loss has to be calculated for both of the predicted bounding boxes) (This isn't explicitely described in paper)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        ) 
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        ) 

        # For class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2), # Converting (N, S, S, 20) -> (N*S*S, 20)
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        return (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
