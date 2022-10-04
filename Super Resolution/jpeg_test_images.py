import cv2
from glob import glob

trainImgFiles=glob('.\\SR_dataset\\Set5_HR\\*.bmp')

origSaveFoler='.\\AR_dataset\\Set5_orig\\'
jpegSaveFoler='.\\AR_dataset\\Set5_jpeg\\'

for trainImgFile in trainImgFiles:
    fileName=trainImgFile.split('\\')[-1]
    origImg=cv2.imread(trainImgFile,cv2.IMREAD_COLOR)
    origImg=cv2.cvtColor(origImg,cv2.COLOR_BGR2GRAY)

    encParam=[int(cv2.IMWRITE_JPEG_LUMA_QUALITY),10]
    _,bitstream = cv2.imencode('.jpeg',origImg,encParam)

    jpegImg= cv2.imdecode(bitstream,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(origSaveFoler+fileName,origImg)
    cv2.imwrite(jpegSaveFoler+fileName,jpegImg)
