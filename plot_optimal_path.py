import os
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

from video_capture import iter_video
from shortest_path import compute_cost_to_go, compute_shortest_path


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    name = args.name

    rootdir = os.path.dirname(__file__)

    # load precomputed image encodings:
    codes1 = np.load(os.path.join(rootdir, "codes", "%s1.npy" % name))
    codes2 = np.load(os.path.join(rootdir, "codes", "%s2.npy" % name))

    # # downsample for the sake of efficiency:
    codes1 = codes1[:, ::5, :]
    codes2 = codes2[:, ::5, :]

    # flatten codes if they aren't already flat:
    vecs1 = np.reshape(codes1, [len(codes1), -1])
    vecs2 = np.reshape(codes2, [len(codes2), -1])

    # compute table of differences:
    norms = np.linalg.norm(vecs1[:, None] - vecs2[None, :], axis=-1)

    # show this table:
    plt.figure(figsize=(8, 6))
    plt.imshow(norms.T, origin="lower", interpolation="nearest")
    plt.title("Image dissimilarities")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # solve the alignment problem:
    cost_to_go = compute_cost_to_go(norms)
    shortest_path = compute_shortest_path(cost_to_go)

    # show this table, now annotated with an optimal path:
    plt.figure(figsize=(8, 6))
    plt.imshow(norms.T, origin="lower", interpolation="nearest")
    plt.plot(*np.transpose(shortest_path), "r-")
    plt.title("Image dissimilarities (with optimal path)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
