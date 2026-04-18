using MIPVerify
using MAT

"""
Load the GTSRB CNNA network from a .mat file exported from PyTorch.

Returns a Sequential network with the following architecture:
- Conv2d(3, 16, 4x4, stride=2, padding=1)
- ReLU
- Conv2d(16, 32, 4x4, stride=2, padding=1)
- ReLU
- Flatten
- Linear(2048 -> 100)
- ReLU
- Linear(100 -> 2)
"""
function load_gtsrb_cnna(weights_path="gtsrb_cnna_weights.mat")
    # Load the local .mat file
    param_dict = matread(weights_path)
    
    # PyTorch format: [out_ch, in_ch, kernel_h, kernel_w]
    # MIPVerify expects: [kernel_h, kernel_w, in_ch, out_ch]
    # Use (3,4,2,1) — not (4,3,2,1), which swaps H/W and misaligns every conv vs PyTorch.
    conv1_weight = permutedims(param_dict["net.0.weight"], (3, 4, 2, 1))
    conv2_weight = permutedims(param_dict["net.2.weight"], (3, 4, 2, 1))
    
    # Convolutional layers (convert bias to 1D vector, specify stride=2 and padding=1)
    conv1 = Conv2d(conv1_weight, vec(param_dict["net.0.bias"]), 2, 1)
    conv2 = Conv2d(conv2_weight, vec(param_dict["net.2.bias"]), 2, 1)
    
    # Fully-connected layers 
    # PyTorch format: [out_features, in_features]
    # MIPVerify expects: [in_features, out_features]
    fc1 = Linear(Matrix(transpose(param_dict["net.5.weight"])), vec(param_dict["net.5.bias"]))
    fc2 = Linear(Matrix(transpose(param_dict["net.7.weight"])), vec(param_dict["net.7.bias"]))
    
    # NHWC (b,h,w,c) -> permute to (w,h,c,b) then vec matches PyTorch NCHW Flatten order (w fastest).
    flat = Flatten([3, 2, 4, 1])

    # Compose the full network
    # Use interval_arithmetic for ALL ReLUs to avoid LP/MIP infeasibility → NaN bounds.
    # (Verification may be looser but avoids "Invalid coefficient NaN" errors.)
    nn = Sequential([
        conv1,
        ReLU(MIPVerify.interval_arithmetic),
        conv2,
        ReLU(MIPVerify.interval_arithmetic),
        flat,
        fc1,
        ReLU(MIPVerify.interval_arithmetic),
        fc2
    ], "GTSRB.CNNA")
    
    return nn
end