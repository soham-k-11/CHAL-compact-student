import os
import numpy as np
from PIL import Image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 130)
    print('|%25s | %100s|' % ('keys', 'values'))
    print('-' * 130)
    for key, value in kwargs.items():
        print('|%25s | %100s|' % (str(key), str(value)))
    print('-' * 130)

def history_frame_paths(path, num_frame):
    dir_path, file_name = os.path.split(path)
    stem, ext = os.path.splitext(file_name)
    index = int(stem)

    paths = [path]
    for id in range(index - 1, index - num_frame, -1):
        for name in ('%0*d' % (len(stem), id), '%d' % id):
            candidate = os.path.join(dir_path, name + ext)
            if id >= 0 and os.path.exists(candidate):
                paths.append(candidate)
                break
        else:
            paths.append(paths[-1])
    return paths[::-1]
