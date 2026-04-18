using MIPVerify
using Gurobi   # use academic license
using Images
using ImageView
using JuMP
using Logging   # to suppress warnings during find_adversarial_example
include("importNN.jl")
include("load_gtsrb_data.jl")
include("masked_perturbation.jl")

function show_rgb(img_4d)
    img = img_4d[1, :, :, :]
    if maximum(img) > 1.01
        img = (img .+ 1) ./ 2
    end
    imshow(colorview(RGB, img[:, :, 1], img[:, :, 2], img[:, :, 3]))
end

#  from MIP Verify repository
function view_diff(diff::Array{<:Real, 2})
    # n = 1001
    # colormap("RdBu", n)[ceil.(Int, (diff .+ 1) ./ 2 .* n)]
    max_val = maximum(abs.(diff))
    max_val == 0 && return RGB.(zeros(size(diff)), zeros(size(diff)), zeros(size(diff)))
    n = diff ./ max_val
    RGB.(max.(n, 0.0), zeros(size(diff)), max.(-n, 0.0))
end


# 0. Pipeline check: display an example test image

# Full test: default "gtsrb_test_data.mat". Eval subset (~390 images): export from notebook (Step 4) then either
#   load_gtsrb_test_data("gtsrb_test_eval.mat") or:  GTSRB_MAT=gtsrb_test_eval.mat julia cnna.jl
println("Loading GTSRB dataset...")
gtsrb = load_gtsrb_test_data("gtsrb_test_eval.mat")
test_set = gtsrb.dataset.test

example_idx = 1
example_img = MIPVerify.get_image(test_set.images, example_idx)
example_label = MIPVerify.get_label(test_set.labels, example_idx)
println("Pipeline check: displaying example test image (index $example_idx, label $example_label; 0=non-stop, 1=stop). Close window to continue.")
show_rgb(example_img)


# 1. Load Custom GTSRB dataset (already loaded above)

@show size(test_set.images)
@show test_set.labels[1:min(10, length(test_set.labels))]


# 2. Load network

println("Loading network n1...")
n1 = load_gtsrb_cnna("gtsrb_cnna_weights01.mat")
println("Network loaded.")


# 3. Fraction of correct predictions on test set

n_test = MIPVerify.num_samples(test_set)
frac_correct = MIPVerify.frac_correct(n1, test_set, n_test)
println("Fraction correct on test set: ", frac_correct)


function binary_confusion_counts(nn, dataset)
    tp = tn = fp = fn = 0  # TP=stop→stop, TN=nonstop→nonstop, FP=nonstop→stop, FN=stop→nonstop
    for i in 1:MIPVerify.num_samples(dataset)
        img = MIPVerify.get_image(dataset.images, i)
        y_true = MIPVerify.get_label(dataset.labels, i)
        y_pred = (img |> nn |> MIPVerify.get_max_index) - 1
        if y_true == 1 && y_pred == 1
            tp += 1
        elseif y_true == 1 && y_pred == 0
            fn += 1
        elseif y_true == 0 && y_pred == 0
            tn += 1
        else
            fp += 1
        end
    end
    return (tp = tp, tn = tn, fp = fp, fn = fn)
end

c = binary_confusion_counts(n1, test_set)
println("Stops (label=1): correct=$(c.tp), wrong=$(c.fn) (recall=$(c.tp / (c.tp + c.fn))))")
println("Non-stops (label=0): correct=$(c.tn), wrong=$(c.fp) (specificity=$(c.tn / (c.tn + c.fp)))")


# 4. Pick a sample image and find one that works for verification
sample_index = nothing
sample_image = nothing
sample_label = nothing
sample_class = nothing
d = nothing

TIME_LIMIT_SEC = 300 #8000
EPSILON = .5
OUT_DIR = "adversarial_results_inf_notimelimit"

mask = sticker_mask((1, 32, 32, 3); row_range=(9, 13), col_range=(15, 22))
# indices_to_try = gtsrb.stop_indices[1:min(30, length(gtsrb.stop_indices))]
indices_to_try = [42]
mkpath(OUT_DIR)
function save_rgb(img_4d, path)
  img = img_4d[1, :, :, :]
  if maximum(img) > 1.01
      img = (img .+ 1) ./ 2
  end
  save(path, colorview(RGB, img[:, :, 1], img[:, :, 2], img[:, :, 3]))
end
found_count = 0
last_original = nothing
last_perturbed = nothing
t_start = time()
for idx in indices_to_try
  global sample_index, sample_image, sample_label, sample_class, d,
         found_count, last_original, last_perturbed
  sample_index = idx
  sample_image = MIPVerify.get_image(test_set.images, sample_index)
  sample_label = MIPVerify.get_label(test_set.labels, sample_index)
  sample_class = sample_label + 1
  target_label_index = sample_class == 2 ? 1 : 2
  try
      println("Trying image $idx (label $sample_label, target $target_label_index, ε=$EPSILON, limit=$(TIME_LIMIT_SEC)s) ...")
      d = @time Logging.with_logger(Logging.NullLogger()) do
          MIPVerify.find_adversarial_example(
              n1,
              sample_image,
              target_label_index,
              Gurobi.Optimizer,
            #   Dict("TimeLimit" => TIME_LIMIT_SEC),
              Dict(),
              norm_order = Inf, #Inf or 1
            #   pp = MIPVerify.LInfNormBoundedPerturbationFamily(EPSILON),
              pp = MaskedLInfNormBoundedPerturbationFamily(EPSILON, mask),
              tightening_algorithm = MIPVerify.interval_arithmetic,
          )
      end
      perturbed_sample_image = JuMP.value.(d[:PerturbedInput])
      perturbation = JuMP.value.(d[:Perturbation])
      save_rgb(sample_image, joinpath(OUT_DIR, "original_image$(idx).png"))
      save_rgb(perturbed_sample_image, joinpath(OUT_DIR, "perturbed_image$(idx).png"))
      save(joinpath(OUT_DIR, "diff_image$(idx).png"), view_diff(perturbation[1, :, :, 1]))
      found_count += 1
      last_original = sample_image
      last_perturbed = perturbed_sample_image
      println("  -> Adversarial found for image $idx; saved to $OUT_DIR/")
  catch e
      println("  Image $idx failed or timed out ($(typeof(e))), skipping.")
      d = nothing
  end
end
println("Done. Found $found_count adversarial examples in $OUT_DIR/")
if found_count > 0
  show_rgb(last_original)
  show_rgb(last_perturbed)
else
  @warn "No adversarial examples found. Try larger EPSILON (e.g. 0.25) or more images."
end
elapsed = time() - t_start
println("Total time: ", round(elapsed, digits=2), " s (", round(elapsed / 60, digits=2), " min)")
