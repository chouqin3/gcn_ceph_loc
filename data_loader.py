import numpy as np
import os
from PIL import Image
import cv2
from math import pi

from torchvision import transforms


class Cephalometric:
    def __init__(self, pathDataset, mode, augment=True, num_landmark=19, resized=[800, 640]):
        self.num_landmark = num_landmark
        self.augment = augment
        self.size = resized
        self.original_size = [2400, 1935]
        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, 'AnnotationsByMD/400_junior')
        self.pth_label_senior = os.path.join(pathDataset, 'AnnotationsByMD/400_senior')
        self.mode = mode
        self.list = list()
        if mode == 'Train':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150  #
        elif mode == 'Test1':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 300
        elif mode == 'Test_all':
            self.pth_Image = os.path.join(self.pth_Image, 'bmpall')
            start = 151
            end = 400
        elif mode == 'Test_img':
            self.pth_Image = os.path.join(self.pth_Image, 'bmpall')
            start = 303
            end = 303
        else:
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400
        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

    def resize_landmark(self, landmark):
        landmark[0] = landmark[0] * self.size[1] / self.original_size[1]
        landmark[1] = landmark[1] * self.size[0] / self.original_size[0]
        return landmark

    def __getitem__(self, index):
        item = self.list[index]
        pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')

        tx = transforms.Compose([
            transforms.Pad((0, 0, 0, 32)),
            transforms.ToTensor(),
            lambda x: x[:, :, :1920].sum(dim=0, keepdim=True), # channel 1
            transforms.Normalize([1.4255656], [0.8835338])
            ])

        item['image'] = np.array(tx(Image.open(pth_img)))[0,:,:]

        M = None
        if self.mode == 'Train' and self.augment:
            tx = transforms.Compose([
                transforms.Pad((0, 0, 0, 32)),
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.3),
                transforms.ToTensor(),
                lambda x: x[:, :, :1920].sum(dim=0, keepdim=True),
                transforms.Normalize([1.4255656], [0.8835338])
            ])

            item['image'] = np.array(tx(Image.open(pth_img)))[0, :, :]
            theta = ((np.random.rand(1) * 2 - 1) * pi * 5 / 180).astype(np.float32)
            trans = ((np.random.rand(2, 1) * 2 - 1) * 10).astype(int)
            scale = (1+(np.random.rand(2,1) * 2 - 1) * 0.01).astype(np.float32)
            rsin = np.sin(theta)
            rcos = np.cos(theta)
            M = np.concatenate((rcos*scale[0], -rsin, trans[0], rsin, rcos*scale[0], trans[1]), axis=0).reshape(2, 3)
            cv2.setNumThreads(0)
            item['image'] = cv2.warpAffine(item['image'], M,
                                           (item['image'].shape[1], item['image'].shape[0]))*((np.random.rand()-0.5)/2.5+1)

        landmark_arr_true = list()
        with open(os.path.join(self.pth_label_junior, item['ID'] + '.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID'] + '.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [0.5 * (int(landmark1[i]) + int(landmark2[i])) for i in range(len(landmark1))]
                    landmark_arr_true.append(list(map(float, self.resize_landmark(landmark.copy()))))

        if M is not None:
            landmark_arr_true = np.dot(np.hstack((np.array(landmark_arr_true), np.ones((19, 1)))), np.transpose(M))
        normalized_ldmk_arr = np.zeros((19, 2))
        for i, landmark in enumerate(landmark_arr_true):
            normalized_ldmk_arr[i] = (np.array(landmark) - 1) / np.array([self.size[1] - 1, self.size[0] - 1])
        middle = np.array([1920, 2432]) / 2
        ty = lambda x: (x - middle) / 1920. * 2
        return item['image'], normalized_ldmk_arr, np.array(landmark_arr_true), ty(landmark_arr_true)

    def __len__(self):
        return len(self.list)


