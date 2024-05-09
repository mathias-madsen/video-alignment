# Deep Video Synchronization

This repository contains some code for aligning two video streams so that similar events happen at similar times when you play them back. The method I use to accomplish this uses [a pretrained neural network](https://pytorch.org/vision/master/models/resnet.html) to measure the similarity between two images, and then a [time-warping](https://en.wikipedia.org/wiki/Dynamic_time_warping) algorithm to align the two sequences.

## Example

Here is a pair of _misaligned_ videos that I created by physically repeating the same sequence of action twice:

https://github.com/mathias-madsen/video-alignment/assets/16747080/a7638ddb-94bd-4376-98e1-e5c9c99ade68

Due to natural variations in my performance, these two videos quickly drift out of sync. However, by using the neural network to annotate each frame and then warping the playback speeds to achieve the best match, we get the following two _aligned_ videos:

https://github.com/mathias-madsen/video-alignment/assets/16747080/d99e264e-f070-4cb6-892d-e5c0516c276d

Note how certain key events (such as the lighting of the match or the extinction of the candles) now happen at exactly the same time.

## Semantic Annotation

This alignment can be accomplished because the pretrained neural network I use is able to make some very sound judgments about which frames are most alike in the two parallel video streams.

We can get a sense of how this network sees the world by seeing what images it considers similar. To do so, we can pick a random frame from one video and then find its most similar match in another video. Here are some examples of such matches:

![supposed_matches](https://github.com/mathias-madsen/video-alignment/assets/16747080/01afc05c-3913-4d46-bb0b-1d77f4a60cd1)

Here again, the two videos recorded separate executions of the same sequence of events. All the left images are taken from the first performance, and all the right images from the second.

In all likelihood, the exact nature of the neural network probably doesn't make much difference, as long as it has some internal representations that respond to semantic and spatial information. (To ensure the latter, I have made added a soft, two-dimensional argmax operation after the last convolutional operation of the network.)

## Framewise Alignment

The actual alignment uses a dynamic-programming algorithm to find out when to pause and when to step the two video streams.

If you think of the cost of displaying two dissimilar images on screen at the same time as a measure of distance, then this algorithm is a shortest-path algorithm. It finds a route that starts at the beginning of both videos and advances to the end of both of them along the path of least dissimilarity:

![shortest_path](https://github.com/mathias-madsen/video-alignment/assets/16747080/2d7f4829-12bf-4597-ba0d-dbbe73569b76)

Note that there are no image pairs that match perfectly (zero dissimilarity), which is what we would expect when physically repeating an action twice in the real world. There is, however, a conspicuous valley of low dissimilarity running through this matrix, and the optimal path rolls along in that valley.

## Baselines

I have compared the time-warping approach to two very rudimentary baselines:

 1. a padding approach where both videos are played at the same speed, and then the last frame just sticks around if one of the two videos ends early
 2. a stretching approach where the speed of the shorter video is slowed down so that the videos have the same duration

Impressionistically, these two baselines seem to do a much worse job than the time warping. Here, for instance, is a pair of videos playing at the same speed:

https://github.com/mathias-madsen/video-alignment/assets/16747080/7f8bcd47-5cf5-4147-8554-047cf5592bd5

Here are the same videos with the speed of the shorter one decreased until they are equally long:

https://github.com/mathias-madsen/video-alignment/assets/16747080/9ab22326-362b-4b25-bea9-0478431de821

And finally, here are the same videos aligned temporally using time warping:

https://github.com/mathias-madsen/video-alignment/assets/16747080/6a052f00-fac1-4f16-8215-71ce1f16ec75

Note how several key events line up in this last pair but not in the previous ones -- for instance, one of the two streams freezes for a relatively long period so that the pouring of the water can happen at the same time.
