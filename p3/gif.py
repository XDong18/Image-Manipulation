from PIL import Image
import os


def make_gif(path, fn):
    file_list = os.listdir(path)
    file_list = sorted(file_list, key= lambda file: int(file.split('.')[0]))
    images = [Image.open(os.path.join(path, file)) for file in file_list]
    images[0].save(fn, format='GIF', append_images=images[1:], save_all=True, duration=33, loop=0)



