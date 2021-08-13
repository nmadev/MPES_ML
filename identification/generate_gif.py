import imageio
import os
from PIL import Image
import glob

# generates a gif from all the images in the single temp random forest folder
paths = os.listdir('./random_forest_id_eval/single')

# generates and saves 3-label confusion matrix gif
images = []
for filename in paths:
    if '3label' in filename:
        images.append(imageio.imread('random_forest_id_eval/single/' + filename))
imageio.mimsave('random_forest_id_eval/3label.gif', images)

# generates and saves 5-label confusion matrix gif
images = []
for filename in paths:
    if '5label' in filename:
        images.append(imageio.imread('random_forest_id_eval/single/' + filename))
imageio.mimsave('random_forest_id_eval/5label.gif', images)