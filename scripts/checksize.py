
from tifffile import imread

import pdb
import os
import tqdm

number_of_rect = 0
number_of_imgs = 0
#for root, dirs, filenames in os.walk("./data/features/projects_sat-io_open-datasets_ASTER_GDEM"):
#for root, dirs, filenames in os.walk("../google_earth_download/projects_sat-io_open-datasets_ASTER_GDEM"):
#for root, dirs, filenames in os.walk("../../google_earth_download/projects_sat-io_open-datasets_ASTER_GDEM"):
#for root, dirs, filenames in os.walk("data/features/projects_sat-io_open-datasets_ASTER_GDEM"):
for root, dirs, filenames in os.walk("data/features/COPERNICUS_S2_SR_HARMONIZED"):
    for f in tqdm.tqdm(filenames):
        img = imread(os.path.join(root,f))
        number_of_imgs += 1
        if img.shape[0] !=  img.shape[1]:
            number_of_rect += 1
            print(f)
            print(f"{number_of_rect} of {number_of_imgs}")
        #print(os.path.join(root,f))
       # pdb.set_trace()
print(number_of_rect)