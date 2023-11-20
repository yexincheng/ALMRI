import numpy as np

def compute_dice_coefficient(mask_gt, mask_pred):
  """Compute soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`. 
  
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
  """
  # assert mask_gt.shape == mask_pred.shape
  assert len(mask_gt.shape)==3 and len(mask_pred.shape)==3, 'should be 3 channels'
  assert mask_gt.shape[0]==mask_pred.shape[0] and mask_gt.shape[1]==mask_pred.shape[1] and mask_gt.shape[2]==mask_pred.shape[2], 'image and ground truth should have the same size'
            
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum