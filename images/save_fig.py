import h5py
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import click


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--big_crop", nargs=4, type=int, default=[0, -1, 0, -1])
@click.option("--dataset_name")
@click.option("--output")
def main(filename, dataset_name, big_crop, output):
    input_file = h5py.File(filename, "r")
    dataset = input_file[dataset_name]
    min_x, max_x, min_y, max_y = big_crop
    print("original shape", dataset.shape)
    dataset = dataset[min_y:max_y, min_x:max_x, ...]
    if dataset.ndim > 2:
        dataset = dataset[..., 0]
    if dataset_name == "postprocessing/visibility":
        median_visibility = np.median(dataset)
        print("median_vis = ", median_visibility)
    plt.figure()
    limits = stats.mstats.mquantiles(
        dataset[dataset < 4e9],
        prob=[0.02, 0.98])
    print(limits)
    image = plt.imshow(
        dataset,
        aspect='auto',
        clim=limits)
    plt.colorbar()
    plt.savefig(
        output,
        bbox_inches="tight",
        dpi=120)



if __name__ == "__main__":
    main()
