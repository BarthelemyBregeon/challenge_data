import numpy as np

def iou_score(y_true, y_pred, num_classes):
    """
    Calculate the Intersection over Union (IoU) score for multiple classes.

    Parameters:
    y_true (array): Ground truth mask with class labels.
    y_pred (array): Predicted mask with class labels.
    num_classes (int): Number of classes (including background).

    Returns:
    dict: IoU score for each class excluding the background.
    """
    iou_scores = {}
    for cls in range(1, num_classes):  # Start from 1 to exclude background
        true_class = (y_true == cls)
        pred_class = (y_pred == cls)
        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()
        if union == 0:
            iou = float('nan')  # Avoid division by zero
        else:
            iou = intersection / union
        iou_scores[f'class_{cls}'] = iou
    return iou_scores

# Example usage
if __name__ == "__main__":

    # Example ground truth and prediction masks
    y_true = np.array([[1, 1, 0, 0],
                       [1, 1, 0, 0],
                       [0, 0, 2, 2],
                       [0, 0, 2, 2]])

    y_pred = np.array([[1, 0, 0, 0],
                       [1, 1, 0, 0],
                       [0, 0, 2, 2],
                       [0, 0, 2, 0]])

    num_classes = 3  # Including background
    scores = iou_score(y_true, y_pred, num_classes)
    for cls, score in scores.items():
        print(f"{cls} IoU score: {score}")