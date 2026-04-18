# GTSRB Stop Sign Adversarial Verification

This project trains a CNN on the GTSRB dataset (binary: stop vs. non-stop signs),
then uses MIPVerify.jl + Gurobi to find provable adversarial examples via mixed-integer programming.

## Workflow 

**Step 1 — Train the model (Python)**
- `gtsrb_stop_sign_CNNA.ipynb` — Jupyter notebook that trains the CNNA model on GTSRB data, evaluates accuracy, and exports the model weights and test data to `.mat` files
- `export_test_data.py` — Helper function for `gtsrb_stop_sign_CNNA.ipynb` to export `gtsrb_test_eval.mat`.
- `requirements.txt`

**Step 2 — Run adversarial verification (Julia)**
- `cnna.jl` — Main code. Loads the network and data, checks accuracy to ensure model loaded correctly, then generates adversarial examples using Gurobi
- `importNN.jl` — Loads `gtsrb_cnna_weights01.mat` and reconstructs the CNN network in MIPVerify format.
- `load_gtsrb_data.jl` — Loads `gtsrb_test_eval.mat` into the MIPVerify dataset format.
- `masked_perturbation.jl` — Limits pixel perturbations to a localized rectangular area

## Data Files

- `gtsrb_cnna_weights01.mat` — Trained network weights 
- `gtsrb_test_eval.mat` — Balanced test set (~390 images) exported from the notebook for Julia evaluation.
- `gtsrb_stop_sign_CNNA.pth` — Raw PyTorch model checkpoint (used to regenerate `.mat` weights if needed).

## Output

- `adversarial_results_*/` — Folders containing saved original, perturbed, and diff images from each verification run.
