from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
import numpy as np

PATH = ".workspace/runs/20221119_autotune_trial_1575_cost_sensitive"
event_acc = EventAccumulator(PATH)
event_acc.Reload()

''' Check the tags under histograms and choose the one you want

Example:
    {
        'images': ['Feature/Embedded', 'Feature/X'],
        'audio': [], 
        'histograms': ['MAPE', 'Label/True/Val', 'Label/Predicted/Val'], 
        'scalars': ['MAPE/Val', 'MAPE/Train', 'Loss/Train', 'Loss/Val'], 
        'distributions': ['MAPE', 'Label/True/Val', 'Label/Predicted/Val'], 
        'tensors': [], 'graph': False, 'meta_graph': False, 'run_metadata': []}
'''
tags = event_acc.Tags()

if sys.argv[1] == "show":
    print(tags)
elif sys.argv[1] == "histograms":
    histograms = event_acc.Histograms(sys.argv[2])
    ### Fetch the histogram of the last step
    event = histograms[-1]
    values = event.histogram_value.bucket_limit
    counts = event.histogram_value.bucket
    original_y = np.exp(values)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 6))
    figbase = 330

    figbase += 1
    ax = fig.add_subplot(figbase)
    plt.title(f"{sys.argv[1].upper()} of {sys.argv[2]} at step {event.step} \n"
        f"of trial {PATH}")
    plt.plot(values, counts)
    plt.xlabel("log(Y)")
    plt.ylabel("Frequency")

    figbase += 1
    ax = fig.add_subplot(figbase)
    plt.plot(original_y, counts)
    plt.xlabel("Y")
    plt.ylabel("Frequency")

    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import QuantileTransformer
    rng = np.random.RandomState(304)
    bc = PowerTransformer(method="box-cox")
    yj = PowerTransformer(method="yeo-johnson")
    # n_quantiles is set to the training set size rather than the default value
    # to avoid a warning being raised by this example
    qt = QuantileTransformer(
        n_quantiles=500, output_distribution="normal", random_state=rng)
    qt2 = QuantileTransformer(
        n_quantiles=500, output_distribution="uniform", random_state=rng)

    _original_y = []
    for _id in range(len(original_y)):
        for _ in range(int(counts[_id])):
            _original_y.append([original_y[_id]])
    _original_y = np.array(_original_y)

    figbase += 1
    ax = fig.add_subplot(figbase)
    plt.hist(bc.fit(_original_y).transform(_original_y), bins=10)
    plt.xlabel("Y with Box-Cox transforms")
    plt.ylabel("Frequency")

    figbase += 1
    ax = fig.add_subplot(figbase)
    plt.hist(yj.fit(_original_y).transform(_original_y), bins=10)
    plt.xlabel("Y with Yeo-Johnson transforms")
    plt.ylabel("Frequency")

    figbase += 1
    ax = fig.add_subplot(figbase)
    # print(original_y.shape)
    plt.hist(qt.fit(_original_y).transform(_original_y), bins=10)
    plt.xlabel("Y with Quantile transform")
    plt.ylabel("Frequency")

    figbase += 1
    ax = fig.add_subplot(figbase)
    # print(original_y.shape)
    import pdb; pdb.set_trace()
    plt.hist(qt2.fit(_original_y).transform(_original_y), bins=10)
    plt.xlabel("Y with Quantile-Uniform transform")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("tmp/tmp2.png")
    plt.close()

else:
    raise ValueError(sys.argv)


# This will give you the list used by tensorboard 
# of the compress histograms by timestep and wall time
# event_acc.CompressedHistograms(HISTOGRAM_TAG)

