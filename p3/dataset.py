import os
import numpy as np 
import json
from utils import mkdir


def ge_detaset_info(path, out_path, out_f, feature='1m', h=480, w=640):
    dataset = []
    corners = [(0, h-1), (w-1, h-1), (0, 0), (w-1, 0)]

    file_list = os.listdir(path)
    for file in file_list:
        if file.split('.')[1]=='asf' and file.split('.')[0][3:5]==feature:
            with open(os.path.join(path, file)) as f:
                lines = f.readlines()

            info_dir = {}
            info_dir['file_name'] = file.split('.')[0] + '.jpg'
            points = []
            for line in lines[16 : 16 + 58]:
                # print(line)
                info = line.split(' \t')
                # print(info)
                points.append([float(info[2])*w, float(info[3])*h])
            points = points + corners
            info_dir['points'] = points
            dataset.append(info_dir)

    mkdir(out_path)
    with open(os.path.join(out_path, feature + '_' + out_f), 'w') as f:
        json.dump(dataset, f)



