import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self):
        inputImgFolder = ".\\AR_dataset\\T91_jpeg"
        labelImgFolder = ".\\AR_dataset\\T91_orig"
        patchSize = 32

        inputImgPaths = glob("%s/*.png" % (inputImgFolder))
        labelImgPaths = glob("%s/*.png" % (labelImgFolder))
        inputImgPaths.sort()
        labelImgPaths.sort()

        self.inputPatchs = []
        self.labelPatchs = []

        for idx in range(len(inputImgPaths)):
            inputImg = cv2.imread(inputImgPaths[idx], cv2.IMREAD_GRAYSCALE)
            labelImg = cv2.imread(labelImgPaths[idx], cv2.IMREAD_GRAYSCALE)
            #
            # inputImg = cv2.resize(inputImg,dsize=(512,512))
            # labelImg = cv2.resize(labelImg,dsize=(512,512))
            inputImg=np.array(inputImg,dtype=np.float32)/255.
            labelImg=np.array(labelImg,dtype=np.float32)/255.

            inputImg = np.expand_dims(inputImg, axis=0)# 이미지 차원 변경
            labelImg = np.expand_dims(labelImg, axis=0)

            self.frameToPatchs(inputImg=inputImg, labelImg=labelImg, patchSize=patchSize)

    def __len__(self):
        return len(self.inputPatchs)

    def __getitem__(self, idx):
        return self.inputPatchs[idx], self.labelPatchs[idx]

    def frameToPatchs(self, inputImg=None, labelImg=None, patchSize=32):
        channel, height, width = labelImg.shape

        numPatchY = height // patchSize
        numPatchX = width // patchSize

        for yIdx in range(numPatchY):
            for xIdx in range(numPatchX):
                xStartPos = xIdx * patchSize
                xFianlPos = (xIdx * patchSize) + patchSize
                yStartPos = yIdx * patchSize
                yFianlPos = (yIdx * patchSize) + patchSize

                self.inputPatchs.append(inputImg[:, yStartPos:yFianlPos, xStartPos:xFianlPos])
                self.labelPatchs.append(labelImg[:, yStartPos:yFianlPos, xStartPos:xFianlPos])


class TestDataset(Dataset):
    def __init__(self):
        inputImgFolder = ".\\AR_dataset\\Set5_jpeg"
        labelImgFolder = ".\\AR_dataset\\Set5_orig"

        inputImgPaths = glob("%s\\*.bmp" % (inputImgFolder))
        labelImgPaths = glob("%s\\*.bmp" % (labelImgFolder))
        inputImgPaths.sort()
        labelImgPaths.sort()

        self.inputImgs = []
        self.labelImgs = []
        self.imgName = []

        for idx in range(len(inputImgPaths)):
            inputImg = cv2.imread(inputImgPaths[idx], cv2.IMREAD_GRAYSCALE)
            labelImg = cv2.imread(labelImgPaths[idx], cv2.IMREAD_GRAYSCALE)
            # inputImg = cv2.resize(inputImg,dsize=(512,512))
            # labelImg = cv2.resize(labelImg,dsize=(512,512))
            inputImg = np.array(inputImg, dtype=np.float32) / 255.
            labelImg = np.array(labelImg, dtype=np.float32) / 255.

            inputImg = np.expand_dims(inputImg, axis=0)  # 이미지 차원 변경
            labelImg = np.expand_dims(labelImg, axis=0)

            self.inputImgs.append(inputImg)
            self.labelImgs.append(labelImg)
            self.imgName.append(inputImgPaths[idx].split("/")[-1])

    def __len__(self):
        return len(self.inputImgs)

    def __getitem__(self, idx):
        return self.inputImgs[idx], self.labelImgs[idx], self.imgName[idx]