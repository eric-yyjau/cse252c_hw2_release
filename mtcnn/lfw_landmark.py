from PIL import Image
from src import detect_faces
import glob
import argparse
import os.path as osp
import numpy as np


def computeArea(points ):
    assert(points.shape[0] == 2 and points.shape[1] == 5)
    x1 = (points[:, 0] - points[:, 4] )
    x2 = (points[:, 1] - points[:, 3] )
    area = np.abs(x1[1] *x2[0] - x1[0] * x2[1] )
    return area

parser = argparse.ArgumentParser(description='PyTorch mtcnn lfw')
parser.add_argument('--lfw', default='../lfw/', type=str)
parser.add_argument('--output', type=str, default = '../sphereFace/data/lfw_landmarkMTCNN.txt')
args = parser.parse_args()

names = glob.glob(osp.join(args.lfw, '*') )
names = [x for x in names if osp.isdir(x) ]
names = sorted(names )
imgs = []
for x in names:
    imgs = imgs + sorted(glob.glob(osp.join(x, '*.jpg') ) )

with open(args.output, 'w') as fOut:
    cnt = 0
    for imgName in imgs:
        cnt += 1
        print('%d/%d: %s' % (cnt, len(imgs), imgName ) )
        img = Image.open(imgName )
        _, landmarks = detect_faces(img )
        faceNum = landmarks.shape[0]
        if faceNum > 1:
            print('Warning: %s faces have been detected!' % faceNum )

        largestMark = np.array(landmarks[0, :] ).reshape([2, 5] )
        largestArea = computeArea(largestMark )
        for n in range(1, faceNum):
            mark = np.array(landmarks[n, :] ).reshape([2, 5] )
            area =  computeArea(mark )
            if area > largestArea:
                largestMark = mark
                largestArea = area

        landmarks = np.transpose(largestMark, [1, 0] ).reshape(10 )

        fOut.write('%s\t' % '/'.join(imgName.split('/')[-2:] ) )
        for f in landmarks:
            fOut.write('%.3f\t' % f )
        fOut.write('\n')

