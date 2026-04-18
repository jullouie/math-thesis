"""
Export GTSRB test data for Julia/MIPVerify
Run this after training model in gtsrb_stop_sign_CNNA.ipynb

I credit AI for helping me develop this file.
"""
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

# Assuming you've already run the notebook and have test_stop_images, test_nonstop_images, etc.
# If running standalone, you'll need to copy the data loading code from the notebook

def export_gtsrb_test_data(test_stop_images, test_stop_labels, 
                           test_nonstop_images, test_nonstop_labels,
                           output_file='gtsrb_test_data.mat'):
    """
    Export GTSRB test data to .mat file for Julia
    
    Args:
        test_stop_images: List of stop sign images (numpy arrays, HWC format)
        test_stop_labels: List of labels (1 for stop)
        test_nonstop_images: List of non-stop sign images
        test_nonstop_labels: List of labels (0 for non-stop)
        output_file: Output .mat filename
    """
    # Combine stop and non-stop images
    all_test_images = test_stop_images + test_nonstop_images
    all_test_labels = test_stop_labels + test_nonstop_labels
    
    # Convert to numpy arrays if not already
    if not isinstance(all_test_images, np.ndarray):
        # Stack images: shape will be (N, H, W, C)
        all_test_images = np.stack(all_test_images, axis=0)
    
    if not isinstance(all_test_labels, np.ndarray):
        all_test_labels = np.array(all_test_labels)
    
    # IMPORTANT: Training in gtsrb_stop_sign_CNNA.ipynb uses [0, 1] (images / 255.0).

    to_neg_one_one = False
    if to_neg_one_one:
        all_test_images = (all_test_images.astype(np.float32) - 0.5) / 0.5
    else:
        all_test_images = all_test_images.astype(np.float32)
    
    # MIPVerify expects: [batch, height, width, channels]
    # our data is already in (N, H, W, C) format
    
    print(f"Test images shape: {all_test_images.shape}")
    print(f"Test labels shape: {all_test_labels.shape}")
    print(f"Test images range: [{all_test_images.min():.3f}, {all_test_images.max():.3f}]")
    print(f"Label distribution: Stop={np.sum(all_test_labels==1)}, Non-Stop={np.sum(all_test_labels==0)}")
    
    # Save to .mat file
    sio.savemat(output_file, {
        'test_images': all_test_images.astype(np.float32),
        'test_labels': all_test_labels.astype(np.int32),
        'stop_indices': np.where(all_test_labels == 1)[0].astype(np.int32),
        'nonstop_indices': np.where(all_test_labels == 0)[0].astype(np.int32)
    })
    
    print(f"\nTest data exported to {output_file}")
    print(f"Total test images: {len(all_test_images)}")
    
    return output_file


def export_gtsrb_eval_tensors(test_images, test_labels, output_file="gtsrb_test_eval.mat"):
    """
    Export the exact (images, labels) you evaluate in PyTorch — e.g. the balanced
    ~390 test set from the notebook — so Julia sees the same tensors as test_accuracy().

    test_images: list of HWC arrays in [0, 1] OR ndarray (N, H, W, C)
    test_labels: list or 1d array of 0/1
    """
    if not isinstance(test_images, np.ndarray):
        test_images = np.stack(test_images, axis=0)
    test_images = test_images.astype(np.float32)
    test_labels = np.asarray(test_labels, dtype=np.int32).ravel()
    print(f"Eval export shape={test_images.shape}, labels={test_labels.shape}")
    print(f"  pixel range [{test_images.min():.4f}, {test_images.max():.4f}] (expect ~[0, 1] for current training)")
    print(f"  Stop={np.sum(test_labels == 1)}, Non-stop={np.sum(test_labels == 0)}")
    sio.savemat(output_file, {
        "test_images": test_images,
        "test_labels": test_labels,
        "stop_indices": np.where(test_labels == 1)[0].astype(np.int32),
        "nonstop_indices": np.where(test_labels == 0)[0].astype(np.int32),
    })
    print(f"Saved {output_file}")
    return output_file

