# James Elgy - 04/07/2023

import numpy as np
from matplotlib import pyplot as plt
from main import main
import os
import string

def overwrite_mur_value(filename, value):
    remove = string.whitespace

    snippet = 'mur=['
    f = open('OCC_Geometry/' + filename, 'r')
    flines = f.readlines()
    for index, line in enumerate(flines):
        place = line.translate(line.maketrans(dict.fromkeys(remove))).find(snippet)
        if place >= 0:
            replacement_characters = str(value)
            tag_start = place
            tag_stop = line[place + 1:].find('-')
            if tag_stop == -1:  # tag is at end of line
                tag_stop = line[place + 1:].find('\n') + 1
            replacement_tag = snippet + replacement_characters + ']'
            replacement_line = line.replace(line[tag_start:tag_start + tag_stop], replacement_tag)

            flines[index] = replacement_line
    with open('OCC_Geometry/' + filename, 'w') as f:
        f.writelines(flines)



if __name__ == '__main__':
    geometry = 'OCC_key_4_July_2023.py'
    m = 200
    s = 100 / 2

    mur = np.random.normal(m, s, 10)

    for inst in mur:
        overwrite_mur_value(geometry, inst)
        print(f'solving for mur = {inst}')
        # for p in [4]:
        #     print(f'solving for order {p}')
        main(geometry=geometry, order=4, use_OCC=True, use_parallel=True, use_POD=True, start_stop=(1,10,161))
