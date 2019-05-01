%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import animation, rc
import cv2
import glob
import sys
import numpy as np
from functools import partial


def play_demo_functional(file_glob_pattern):
    f_names = sorted(glob.glob(file_glob_pattern))
    # print(f_names)

    ims = [cv2.imread(f_name) for f_name in f_names]
    X = np.linspace(0, np.pi * 2.0, len(ims))


    fig, (ax1, ax2) = plt.subplots(1,2)
    drawn_line = ax2.plot(X, np.sin(X))[0]
    drawn_im = ax1.imshow(ims[0][110:-100, 20:-50, [2,1,0]], animated=True)

    def ani_func(i):
        drawn_line.set_data(X[:i], np.sin(X)[:i])
        drawn_im.set_data(ims[i][110:-100, 20:-50, [2,1,0]])
        return drawn_line, drawn_im

    return animation.FuncAnimation(fig, ani_func, interval=100, frames=len(ims), blit=True)
    #plt.show()




def play_demo(file_glob_pattern):
    X = np.linspace(0, 2*np.pi, 100)
    Y = np.sin(X)

    f_names = sorted(glob.glob(file_glob_pattern))
    print(f_names)

    fig, (ax1, ax2) = plt.subplots(1,2)
    ims = []

    ims = (cv2.imread(f_name) for f_name in f_names)
    im_arts = [ax1.imshow(im[110:-100, 20:-50, [2, 1, 0]], animated=True) for im in ims]
    sin_arts = [ax2.plot(X, np.sin(X + t * 0.1), animated=True)[0] for t in range(len(im_arts))]
    full_arts = [list(a) for a in zip(im_arts, sin_arts)]
    print(full_arts[0])


    ani = animation.ArtistAnimation(fig, full_arts, interval=100, blit=True, repeat_delay=1000)

    plt.show()


ani = play_demo_functional("demos/test/Demo_2019-04-25 17:12:24.045983/*.jpg")
from IPython.display import HTML
HTML(ani.to_jshtml())

# if __name__ == "__main__":
#   if len(sys.argv) != 2:
#       print("Usage: python play_demo.py <file_glob_pattern>")
#       sys.exit(0)

#   print(sys.argv)
#   play_demo_functional(sys.argv[1])