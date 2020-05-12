import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
from torch.utils.data import Dataset
from matlab_cp2tform import get_similarity_transform_for_PIL

class BatchLoader(Dataset ):
    def __init__(self, imageRoot = '../CASIA-WebFace/',
            alignmentRoot = './data/casia_landmark.txt', cropSize = (96, 112) ):
        super(BatchLoader, self).__init__()

        self.imageRoot = imageRoot
        self.alignmentRoot = alignmentRoot
        self.cropSize = cropSize
        refLandmark = [ [30.2946, 51.6963],[65.5318, 51.5014],
                [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        self.refLandmark = np.array(refLandmark, dtype = np.float32 ).reshape(5, 2)

        with open(alignmentRoot, 'r') as labelIn:
            labels = labelIn.readlines()

        self.imgNames, self.targets, self.landmarks = [], [], []
        for x in labels:
            xParts = x.split('\t')
            self.imgNames.append(osp.join(self.imageRoot, xParts[0] ) )
            self.targets.append(int(xParts[1] ) )
            landmark = []
            for n in range(0, 10):
                landmark.append(float(xParts[n+2] ) )
            landmark = np.array(landmark, dtype=np.float32 )
            self.landmarks.append(landmark.reshape(5, 2) )

        self.count = len(self.imgNames )
        self.perm = list(range(self.count ) )
        random.shuffle(self.perm )

    def __len__(self):
        return self.count

    def __getitem__(self, ind ):

        imgName = self.imgNames[self.perm[ind] ]
        landmark = self.landmarks[self.perm[ind] ]
        target = np.array([self.targets[self.perm[ind] ] ], dtype=np.int64 )

        # Align the image
        img = Image.open(imgName )
        img = self.alignment(img, landmark )
        img = (img.astype(np.float32 ) - 127.5) / 128

        batchDict = {
                'img': img,
                'target': target
                }
        return batchDict


    def alignment(self, img, landmark ):
        tfm = get_similarity_transform_for_PIL(landmark, self.refLandmark.copy() )
        img = img.transform(self.cropSize, Image.AFFINE,
                tfm.reshape(6), resample=Image.BILINEAR)
        img = np.asarray(img )
        if len(img.shape ) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        else:
            img = img[:, :, ::-1]

        img = np.transpose(img, [2, 0, 1] )
        return img
