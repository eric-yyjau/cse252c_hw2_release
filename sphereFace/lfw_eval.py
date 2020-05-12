import torch
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import argparse
import numpy as np
import os.path as osp
from PIL import Image
from matlab_cp2tform import get_similarity_transform_for_PIL
import faceNet

def alignment(src_img, src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_PIL(s, r)

    face_img = src_img.transform(crop_size, Image.AFFINE,
            tfm.reshape(6), resample=Image.BILINEAR)
    face_img = np.asarray(face_img )[:, :, ::-1]

    return face_img

def cropping(src_img ):
    # IMPLEMENT cropping the center of image







    return src_img

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i*n/n_folds) : int((i+1)*n/n_folds ) ]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='faceNet', type=str)
parser.add_argument('--lfw', default='../lfw/', type=str)
parser.add_argument('--alignmentMode', type=int, default=1,
        help='0: crop the center region of the image, 1: do alignment use provided landmarks 2: do alignment use predicted landmarks')
parser.add_argument('--model','-m', default='./model/sphere20a_20171020.pth', type=str)
args = parser.parse_args()

predicts=[]
net = getattr(faceNet, args.net)()
net.load_state_dict(torch.load(args.model) )
net.cuda()
net.eval()
net.feature = True

landmark = {}
if args.alignmentMode <= 1:
    landMarkFile = osp.join('data', 'lfw_landmark.txt')
else:
    landMarkFile = osp.join('data', 'lfw_landmarkMTCNN.txt')

with open(landMarkFile ) as f:
    landmark_lines = f.readlines()

for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0]] = [float(k) for k in l[1:] if len(k) > 0]

with open('data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]

pairNum = len(pairs_lines )
for i in range(pairNum ):
    print('Process %d/%d' % (i, pairNum ) )
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))

    imgName1 = osp.join(args.lfw, name1 )
    imgName2 = osp.join(args.lfw, name2 )
    img1 = Image.open(imgName1 )
    img2 = Image.open(imgName2 )
    if args.alignmentMode != 0:
        img1 = alignment(img1, landmark[name1] )
        img2 = alignment(img2, landmark[name2] )
    else:
        img1 = cropping(img1 )
        img2 = cropping(img2 )

    imglist = [img1, img2 ]
    for i in range(len(imglist)):
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[i] = (imglist[i]-127.5)/128.0

    img = np.vstack(imglist)
    img = Variable(torch.from_numpy(img).float() ).cuda()
    output = net(img)
    f = output.data
    f1,f2 = f[0],f[1]
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))


accuracy = []
thd = []
folds = KFold(n=pairNum, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array([*map(lambda line:line.strip('\n').split(), predicts) ] )
for idx, (train, test) in enumerate(folds):
    best_thresh = find_best_threshold(thresholds, predicts[train] )
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
