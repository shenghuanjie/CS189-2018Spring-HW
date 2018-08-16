from skimage.io import imsave
from skimage.transform import resize
from os import listdir
from os.path import isfile, join
from rawpy import imread

mypath = 'D:\Chrome\Camera\dng'
savepath = 'D:\Chrome\Camera\data'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for _, filename in enumerate(onlyfiles):
    pure_name = filename[:-4]
    img = imread(mypath + '\\' + filename).postprocess()
    height, width, _ = img.shape
    start = int(width/2-height/2)
    # img = img[:, start:start+height]
    img = resize(img, (1000, 1000), mode='reflect')
    imsave(savepath + '\\' + pure_name + '.png', img)
