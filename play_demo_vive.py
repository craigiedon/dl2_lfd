import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import cv2
import glob
import sys



def play_demo(img_glob_pattern):
    # Specify a folder, then search for each of the images in this 
    f_names = sorted(glob.glob(img_glob_pattern))
    print('Num Files:', len(f_names))

    fig = plt.figure()
    ims = []


    ims = (cv2.imread(f_name) for f_name in f_names)
    im_arts = [[plt.imshow(im[:, :, [2, 1, 0]], animated=True)] for im in ims]

    # Make cleaner by removing axis and whitespace
    plt.axis('off')
    plt.tight_layout()
    # plt.margins(0,0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ani = animation.ArtistAnimation(fig, im_arts, interval=15, blit=True, repeat_delay=50)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='pc'), bitrate=1800)
    ani.save('video.mp4', writer=writer)

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python play_demo.py <file_glob_pattern>")
        sys.exit(0)
    print('Args:', sys.argv)
    img_glob_pattern = (sys.argv[1])
    play_demo(img_glob_pattern)
