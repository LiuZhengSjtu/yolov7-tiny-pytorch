import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image
from skimage import measure


def BottleneckV1(in_channels, out_channels, stride):
  return  nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=stride,padding=1,groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV1, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            BottleneckV1(32, 64, stride=1),
            BottleneckV1(64, 128, stride=2),
            BottleneckV1(128, 128, stride=1),
            BottleneckV1(128, 256, stride=2),
            BottleneckV1(256, 256, stride=1),
            BottleneckV1(256, 512, stride=2),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 1024, stride=2),
            BottleneckV1(1024, 1024, stride=1),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.linear = nn.Linear(in_features=1024,out_features=num_classes)
        # self.linear2 = nn.
        self.dropout = nn.Dropout(p=0.2)
        # self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.ReLU()

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bottleneck(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        x = self.linear(x)
        out = self.softmax(x)
        return out

def myTrain():
    model = MobileNetV1(224)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    lossfun = nn.MSELoss()
    print(model)
    
    for i in range(1000):
        input = torch.randn(1, 3, 224, 224)
        inputlab = torch.mean(input, dim=2)
        inputlab = torch.mean(inputlab, dim=1)
        inputlab = inputlab * 2 + 6
        # inputlab = torch.squeeze(inputlab, dim=0)
        # print(inputlab.shape)
        
        optimizer.zero_grad()
        out = model(input)
        loss = lossfun(out * 10, inputlab)
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            print(i, '  loss :', loss.item())
    
    print(out.shape)

# def datalist():
#     #   read the image to the list, split the images to train, test and eval sets.
#     train_rate = 0.7
#     test_rate = 0.15
#     imgpath = './nyu_hand_dataset_rdf/dataset_rdf/dataset/'
#     imglist = glob.glob(imgpath + '*.png')
#     imglist_rd = imglist
#     random.seed(1)
#     random.shuffle(imglist_rd)
#     imglist_train = imglist_rd[0:int(train_rate * len(imglist))]
#     imglist_test  = imglist_rd[int(train_rate * len(imglist)):int((test_rate + test_rate) * len(imglist))]
#     imglist_eval = imglist_rd[int((test_rate + test_rate) * len(imglist)):]
#     return imglist_train, imglist_test, imglist_eval

# class TransfomrsImage():
#     def __init__(self):
#


class ImageLoad():
    def __init__(self,path = '/homeL/zheng/PycharmProjects/yolov7-tiny-pytorch/nyu_hand_dataset_rdf/dataset_rdf/dataset/'):
                #   /homeL/zheng/PycharmProjects/yolov7-tiny-pytorch/nyu_hand_dataset_rdf/dataset_rdf/dataset/
        self.train_rate = 0.8
        self.test_rate = 0.1
        self.trainread = 0
        self.testread = 0
        self.evalread = 0
        self.resizelen = 800
        self.filepath = path
        self.imglist0 = glob.glob(self.filepath + '*.png')
        self.listlen = len(self.imglist0)
        self.randomimg()
        self.trainmode = True        #   false for test

        self.TransformsImags = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30, fill=0),
            transforms.RandomAffine(degrees=0, scale=(0.8,1.2),shear=45,fill=0)
        ])
        
    def randomimg(self):
        self.imglist = self.imglist0
        random.seed(1)
        random.shuffle(self.imglist)
        
        self.imglist_train = self.imglist[0:int(self.train_rate * self.listlen)]
        self.imglist_test = self.imglist[int(self.train_rate * self.listlen):int((self.train_rate + self.test_rate) * self.listlen)]
        self.imglist_eval = self.imglist[int((self.train_rate + self.test_rate) * self.listlen):]
        self.trainlistlen = len(self.imglist_train)
        self.testlistlen = len(self.imglist_test)
        self.evallistlen = len(self.imglist_eval)
        
    def getimage(self,Aug=True,Show=False,Imgnum = None,Mode=True,Predict=False):
        if Predict:
            outimgpath = self.imglist_eval[self.evalread]
            self.evalread += 1
            if self.evalread >= self.evallistlen:
                self.evalread = 0
            # outimgpath = '/homeL/zheng/PycharmProjects/yolov7-tiny-pytorch/nyu_hand_dataset_rdf/dataset_rdf/dataset/depth_0002054.png'      #   no hand
            # print(outimgpath)
        else:
            if Mode:
                outimgpath = self.imglist_train[self.trainread]
                self.trainread += 1
                if self.trainread >= self.trainlistlen:
                    self.trainread = 0
            else:
                outimgpath = self.imglist_test[self.testread]
                self.testread += 1
                if self.testread >= self.testlistlen:
                    self.testread = 0

        if Imgnum == 0:
            outimgpath = './nyu_hand_dataset_rdf/dataset_rdf/dataset\\depth_0002054.png'
        elif Imgnum == 1:
            outimgpath = './nyu_hand_dataset_rdf/dataset_rdf/dataset\\depth_0002934.png'
        elif Imgnum == 2:
            outimgpath = './nyu_hand_dataset_rdf/dataset_rdf/dataset\\depth_0005228.png'
            
            

        img0 = Image.open(outimgpath)
        
        self.origin_h = img0.height
        self.origin_w = img0.width
        return self.Aug(img0,Aug,Show,Predict=Predict)
        
    def Aug(self,img0, aug=True,Show=False,Predict=False):
        img0_res = np.asarray(img0)
        fillc = img0_res[0, 0, 2] + img0_res[0, 0, 1] * 256  # fill background by original image background
        self.imgdepth = []
        self.imglabel = []

        if aug == True and (not Predict):
            img_aug = self.transforms(img0)
            #   divide to each channel, split to depth image and label
            img_res = np.asarray(img_aug)
            self.imgdepth = img_res[:, :, 2] + img_res[:, :, 1] * 256  # depth
            # fill the expanded area with background
            self.imgdepth[np.where(self.imgdepth == 0)] = fillc
            self.imglabel = img_res[:, :, 0]  # label
        else:
            self.imgdepth = img0_res[:, :, 2] + img0_res[:, :, 1] * 256
            self.imglabel = img0_res[:, :, 0]
        
        self.image_depth = self.imgdepth                     #   800*800 for train and test, in eval it depends on practical img size
        self.image_label = self.imglabel                    #   800*800
        self.arr_label = self.getconnectarea()            #   2*7
        
        self.imgcrop()          #   self.image_depth_crop, self.image_label_crop
        
        if Show == True:
            self.showimg(self.image_depth,self.imglabel,self.arr_label)
            self.showimg(self.image_depth_crop,self.image_label_crop  ,self.arr_label_crop)
 
        #   return depth image and center (wx, hy) of hand rectangle
        return self.image_depth_crop, self.arr_label_crop
    
    def showimg(self,depthimg,labelimg,arr_label):
        '''
        display m*n array by a gray pic, annotate the pic by image_label
        :param depthimg:
        :param image_label:
        :return:
        '''
        # #  ------------------ for image display  -----------------------------
        img_ori = (depthimg - np.min(depthimg)) / (np.max(depthimg) - np.min(depthimg)) * 255
        img_origin = img_ori.astype('uint8')
        cv2.imshow('origin depth iamge', img_origin)
    
        img_lab = labelimg.astype('uint8')
        hand_idx = np.where(img_lab > 200)
        # cv2.rectangle(img_lab, (np.min(hand_idx[1]), np.min(hand_idx[0])), (np.max(hand_idx[1]), np.max(hand_idx[0])), (255, 0, 0))
        
        for i in range(2):
            if arr_label[i][0] > 1:
                cv2.circle(img_lab, (arr_label[i][2], arr_label[i][1]), 10*i+15,(155, 0, 0),-1)
                cv2.rectangle(img_lab, (arr_label[i][4], arr_label[i][3]),(arr_label[i][6], arr_label[i][5]), (255, 0, 0))
            
        cv2.imshow('image lable area', img_lab)
        cv2.waitKey(0)
        # print(img_lab.shape)

    def transforms(self,x):
        '''
        conduct a resize to enlarge the image x, then conduct rotation,flip,affine
        :param x:
        :return:
        '''
        pt = int((self.resizelen - self.origin_h)*0.5)
        pb =  self.resizelen - self.origin_h - pt
        pl = int((self.resizelen - self.origin_w) * 0.5)
        pr = self.resizelen - self.origin_w - pl
        y = transforms.Pad((pl,pt,pr,pb))(x)
        self.fpt = float(pt) / float(self.resizelen)
        self.fpl = float(pl) / float(self.resizelen)
        return self.TransformsImags(y)
    
    def getconnectarea(self):
        '''
        regarding the label image, turn to binary image, find connected areas
        :return: the area, center location of each connected area
        '''
        img_lab = self.imglabel.astype('uint8')
        img_lab_bin = np.uint8(img_lab>0)*255
        # hand_idx = np.where(img_lab > 200)
        labels = measure.label(img_lab_bin, connectivity=2)
        properties = measure.regionprops(labels)
        areanum = properties.__len__()
        out = np.zeros((2,7),dtype=np.uint32)                       #   at most, two hands are detected
        if areanum > 0 :
            for prop in properties:
                if (prop.area > 10) :
                    if prop.area > out[0][0]:
                        out[1] = out[0]
                        rownum = 0
                    elif prop.area > out[1][0]:
                        rownum = 1
                    else:
                        continue
                        
                    out[rownum] = np.array([prop.area , int(prop.centroid[0]) , int(prop.centroid[1]), np.min(prop.coords[:,0]), np.min(prop.coords[:,1]), np.max(prop.coords[:, 0]) ,np.max(prop.coords[:, 1])])
                        # out[0][0] = prop.area                #   area of each connected area
                        # out[0][1] = int(prop.centroid[0])     #   pixel distance to top
                        # out[0][2] = int(prop.centroid[1])     #   pixel distance to left
                        # out[0][3] = np.min(prop.coords[:,0])
                        # out[0][4] = np.min(prop.coords[:,1])
                        # out[0][5] = np.max(prop.coords[:, 0])
                        # out[0][6] = np.max(prop.coords[:, 1])
        return out
    
    def imgcrop(self,len=416):
        span =np.array(self.image_depth.shape)  - len
        croplt = (span * np.random.rand(2)).astype(np.int32)
        self.image_depth_crop = self.image_depth[croplt[0]:croplt[0] + len, croplt[1]:croplt[1] + len]
        self.image_label_crop = self.image_label[croplt[0]:croplt[0] + len, croplt[1]:croplt[1] + len]
        croplab = np.array([[0,croplt[0],croplt[1],croplt[0],croplt[1],croplt[0],croplt[1]],[0,croplt[0],croplt[1],croplt[0],croplt[1],croplt[0],croplt[1]]])
        self.arr_label_crop = np.clip( self.arr_label - croplab,0,len-1)
        for i in range(2):
            if self.arr_label_crop[i][3] == self.arr_label_crop[i][5] or self.arr_label_crop[i][4] == self.arr_label_crop[i][6]:
                self.arr_label_crop[i][0] = 0
        # print(self.arr_label_crop.shape)
                
        # return self.image_depth_crop, self.image_label_crop
        
imghandle = ImageLoad()
haha=0


#
# if __name__=='__main__':
#     # imglist_train, imglist_test, imglist_eval = datalist()
#     # img_depth, img_label = loaddata(imglist_train[0])
#     # img_depth, img_label = loaddata('./nyu_hand_dataset_rdf/dataset_rdf/dataset\\depth_0003342.png')
#     imghandle = ImageLoad()
#     for i in range(4):
#         depth, label = imghandle.getimage(Aug=True,Show=True,Imgnum=2)
#     myTrain()
