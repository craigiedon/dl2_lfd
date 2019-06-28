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
    f_names = sorted(glob.glob(file_glob_pattern))

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ims = []

    ims = (cv2.imread(f_name) for f_name in f_names)
    im_arts = [[ax1.imshow(im[110:-100, 20:-50, [2, 1, 0]], animated=True)] for im in ims]


    ani = animation.ArtistAnimation(fig, im_arts, interval=12, blit=True, repeat_delay=1000)
    return ani



ani = play_demo("/home/cinnes/visual_servo_induction/visuomotor_data/*.jpg")
ani.save("demo.mp4")
print("Saved")
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# if __name__ == "__main__":
#   if len(sys.argv) != 2:
#       print("Usage: python play_demo.py <file_glob_pattern>")
#       sys.exit(0)

#   print(sys.argv)
#   play_demo_functional(sys.argv[1])