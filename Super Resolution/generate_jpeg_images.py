import cv2
from glob import glob

trainImgFiles=glob('.\\SR_dataset\\T91_HR\\*.png')

origSaveFoler='.\\AR_dataset\\T91_orig\\'
jpegSaveFoler='.\\AR_dataset\\T91_jpeg\\'

for trainImgFile in trainImgFiles:
    fileName=trainImgFile.split('\\')[-1]

    origImg=cv2.imread(trainImgFile,cv2.IMREAD_COLOR)
    origImg=cv2.cvtColor(origImg,cv2.COLOR_BGR2GRAY)

    encParam=[int(cv2.IMWRITE_JPEG_LUMA_QUALITY),10]
    _,bitstream = cv2.imencode('.jpeg',origImg,encParam)

    jpegImg= cv2.imdecode(bitstream,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(origSaveFoler+fileName,origImg)
    cv2.imwrite(jpegSaveFoler+fileName,jpegImg)
