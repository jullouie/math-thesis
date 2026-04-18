using MAT
using MIPVerify

"""
Load GTSRB test data in MIPVerify format.

Uses the same dataset types as `MIPVerify.read_datasets()`:
- `LabelledImageDataset`: images 4D (num_samples, height, width, num_channels), labels 1D
- `NamedTrainTestDataset`: name + train + test (train is empty since we only export test)

Returns a named tuple:
- `dataset`: NamedTrainTestDataset — use `dataset.test` like `mnist.test` (same API as read_datasets)
- `stop_indices`: indices of stop sign images (1-based)
- `nonstop_indices`: indices of non-stop sign images (1-based)

Usage (matches MIPVerify.read_datasets API):
    gtsrb = load_gtsrb_test_data()
    # Same as with MNIST:
    sample_image = MIPVerify.get_image(gtsrb.dataset.test.images, 1)
    sample_label = MIPVerify.get_label(gtsrb.dataset.test.labels, 1)
    frac_correct(nn, gtsrb.dataset.test, MIPVerify.num_samples(gtsrb.dataset.test))
"""
function load_gtsrb_test_data(data_path="gtsrb_test_data.mat")
    data = matread(data_path)

    # Images: 4D (num_samples, height, width, channels) — required by LabelledImageDataset
    test_images = data["test_images"]
    test_labels = vec(data["test_labels"])

    # Ensure types match what MIPVerify expects (Real images, Integer labels)
    test_images = Float32.(test_images)
    test_labels = Int32.(test_labels)

    # PyTorch training uses [0, 1] (e.g. /255). If the .mat was saved as [-1, 1],
    # the net will behave like "always non-stop" in Julia; map back to [0, 1].
    if minimum(test_images) < -0.01f0
        test_images = (test_images .+ 1f0) ./ 2f0
        println("  (Remapped test images from [-1, 1] to [0, 1] to match training.)")
    end

    # Build test set as LabelledImageDataset (same type as mnist.test)
    test = MIPVerify.LabelledImageDataset(test_images, test_labels)

    # NamedTrainTestDataset expects train + test; we only have test, so use empty train
    n_samples, h, w, c = size(test_images)
    empty_train = MIPVerify.LabelledImageDataset(
        zeros(Float32, 0, h, w, c),
        Int32[]
    )
    dataset = MIPVerify.NamedTrainTestDataset("GTSRB", empty_train, test)

    # Optional: indices for stop vs non-stop (Julia 1-based)
    stop_indices = haskey(data, "stop_indices") ? (vec(data["stop_indices"]) .+ 1) : findall(==(1), test_labels)
    nonstop_indices = haskey(data, "nonstop_indices") ? (vec(data["nonstop_indices"]) .+ 1) : findall(==(0), test_labels)

    println("GTSRB data loaded (MIPVerify format):")
    println("  dataset.test: ", test)  # Only show test; showing full dataset would access empty train.images[1]
    println("  Stop indices: ", length(stop_indices), " | Non-stop indices: ", length(nonstop_indices))
    println("  test_images extrema (after load/remap): ", extrema(test_images), " — training uses [0, 1]")

    return (; dataset, stop_indices, nonstop_indices)
end

"""
Get a random stop sign from the test set.
Returns (image, label, index). Use with result of load_gtsrb_test_data().
"""
function get_random_stop_sign(gtsrb_data)
    idx = rand(gtsrb_data.stop_indices)
    image = MIPVerify.get_image(gtsrb_data.dataset.test.images, idx)
    label = MIPVerify.get_label(gtsrb_data.dataset.test.labels, idx)
    return image, label, idx
end

"""
Get a random non-stop sign from the test set.
Returns (image, label, index).
"""
function get_random_nonstop_sign(gtsrb_data)
    idx = rand(gtsrb_data.nonstop_indices)
    image = MIPVerify.get_image(gtsrb_data.dataset.test.images, idx)
    label = MIPVerify.get_label(gtsrb_data.dataset.test.labels, idx)
    return image, label, idx
end
