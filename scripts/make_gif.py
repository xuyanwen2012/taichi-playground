import imageio
from os import listdir, remove
from os.path import isfile, join, dirname
import numpy as np

PATH = join(dirname(__file__), '../nbody_out')

images = [f for f in listdir(PATH) if isfile(join(PATH, f))]

raw_images = [f for f in images if '_plt' not in f]
plt_images = [f for f in images if '_plt' in f]


def make_gif(out_name, source_pngs):
    ret = out_name + '.gif'
    with imageio.get_writer(ret, mode='I') as writer:
        for filename in source_pngs:
            image = imageio.imread(join(PATH, filename))
            writer.append_data(image)
    return ret


def concatenate_gifs(lhs, rhs):
    gif1 = imageio.get_reader(lhs)
    gif2 = imageio.get_reader(rhs)
    # If they don't have the same number of frame take the shorter
    number_of_frames = min(gif1.get_length(), gif2.get_length())

    # Create writer object
    new_gif = imageio.get_writer('output.gif')

    for frame_number in range(number_of_frames):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        # here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()


lhs = make_gif("mygif1", raw_images)
rhs = make_gif("mygif2", plt_images)
concatenate_gifs(lhs, rhs)
remove(lhs)
remove(rhs)
