"""
Save a movie showing the aligned videos with frame numbers.

Usage:

>>> python3 process_videos.py <NAME> <MATCHING>

where <NAME> identifies two .mov files in the videos/ folder, e.g.,

    videos/candle1.mov
    videos/candle2.mov

and <MATCHING> is one of

 - optimal : use time warp to play back the videos in the way that
             achieves the highest level of frame-pair similarity.
 - pad : play back both videos at their natural speed, but let the
         last frame of the shortest video hang around until the
         longest video is finished.
 - stretch : adjust the speed of the shortest video so that that it
             has the same length as the longest, without taking the
             video contents into account.
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from argparse import ArgumentParser

import shortest_path
from video_capture import iter_video


parser = ArgumentParser()
parser.add_argument("name")
parser.add_argument("matching",
                    default="optimal",
                    choices=["optimal", "pad", "stretch"])


if __name__ == "__main__":

    args = parser.parse_args()
    name = args.name
    matching = args.matching

    rootdir = os.path.dirname(__file__)

    # load precomputed image encodings:
    codes1 = np.load(os.path.join(rootdir, "codes", "%s1.npy" % name))
    codes2 = np.load(os.path.join(rootdir, "codes", "%s2.npy" % name))

    # # downsample for the sake of efficiency:
    codes1 = codes1[:, ::5]
    codes2 = codes2[:, ::5]

    # flatten codes if they aren't already flat:
    vecs1 = np.reshape(codes1, [len(codes1), -1])
    vecs2 = np.reshape(codes2, [len(codes2), -1])

    # compute table of differences and solve the alignment problem:
    norms = np.linalg.norm(vecs1[:, None] - vecs2[None, :], axis=-1)
    cost_to_go = shortest_path.compute_cost_to_go(norms)
    optimal_path = shortest_path.compute_shortest_path(cost_to_go)

    # load the videos:
    vidpath1 = os.path.join(rootdir, "videos", "%s1.mov" % name)
    vidpath2 = os.path.join(rootdir, "videos", "%s2.mov" % name)
    print("Loading videos . . .")
    video1 = np.array(list(iter_video(vidpath1)))
    video2 = np.array(list(iter_video(vidpath2)))
    print("Done loading videos.\n")

    # get a path from start to finish of the requested type:
    if matching.lower().startswith("optimal"):
        path = optimal_path
    elif matching.lower().startswith("stretch"):
        path = shortest_path.compute_straight_path(len(video1), len(video2))
    elif matching.lower().startswith("pad"):
        path = shortest_path.compute_hinged_path(len(video1), len(video2))
    else:
        raise ValueError("Unexpected matching scheme: %r" % matching)

    # set up the plot:
    figure, (top, bot) = plt.subplots(ncols=2, figsize=(12, 4))
    top.axis("off")
    bot.axis("off")
    topimshow = top.imshow(video1[0])
    botimshow = bot.imshow(video2[0])
    top.set_title("frame 1/%s" % len(video1))
    bot.set_title("frame 1/%s" % len(video2))
    figure.tight_layout()
    num_frames = max(len(video1), len(video2))
    progress_bar = tqdm(total=num_frames)
    
    def update(step):
        i, j = path[step]
        if i < len(video1):
            topimshow.set_data(video1[i])
            top.set_title("frame %s/%s" % (i, len(video1)))
        if j < len(video2):
            botimshow.set_data(video2[j])
            bot.set_title("frame %s/%s" % (j, len(video2)))
        progress_bar.update(1)

    animation = FuncAnimation(fig=figure,
                              func=update,
                              frames=num_frames,
                              interval=1000 / 30)

    filename = "%s_%s.mp4" % (name, matching)
    filepath = os.path.join(rootdir, "animations", filename)

    print("Saving animation . . .")
    os.makedirs(os.path.join(rootdir, "animations"), exist_ok=True)
    animation.save(filename=filepath, writer="ffmpeg")
    progress_bar.close()
    print("Done saving animation.\n")
