from ImgProcessing import ImgProcessing
from MF import MF
if __name__ == '__main__':
    original_pixels = ImgProcessing.image_load("./Lena_00c.png")
    
    corruption_rate = 60.0
    corrupted_pixels=ImgProcessing.image_load("./Lena_60c.png")
    corrupted_pixels=corrupted_pixels / 255
    
    zero_pixels=0
    minus_pixels=0
    plus_pixels=0
    original_shape=original_pixels.shape
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            value=corrupted_pixels[i][j]
            if value==0:
                zero_pixels+=1
            elif value<0:
                minus_pixels+=1
            else:
                plus_pixels+=1
                
    print('zero: %d, minus: %d, plus: %d'%(zero_pixels,minus_pixels,plus_pixels))
    mf=MF(corrupted_pixels,64,0.1,0.01,100,True)
    
    print('Train_start')
    mf.train()
    
    reconsted=mf.get_Matrix()
    ImgProcessing.print_img_list(original_pixels,corrupted_pixels,reconsted,corruption_rate)