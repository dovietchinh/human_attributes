import numpy as np 
import cv2
import random
def preprocess(img,img_size,padding=True):
    """[summary]

    Args:
        img (np.ndarray): images 
        img_size (int,list,tuple): target size. eg: 224 , (224,224) or [224,224]
        padding (bool): padding img before resize. Prevent from image distortion. Defaults to True.

    Returns:
        images (np.ndarray): images in target size
    """
    if padding:
        height,width,_ = img.shape 
        delta = height - width 
        
        if delta > 0:
            img = np.pad(img,[[0,0],[delta//2,delta//2],[0,0]], mode='constant',constant_values =255)
        else:
            img = np.pad(img,[[-delta//2,-delta//2],[0,0],[0,0]], mode='constant',constant_values =255)
    if isinstance(img_size,int):
        img_size = (img_size,img_size)
    return cv2.resize(img,img_size)

class RandAugment:

    def __init__(self, augment_params):
        self.num_layers = augment_params['num_layers']       
        self.AUGMENT_FUNCTION = {
            'fliplr' : RandAugment.augment_fliplr if augment_params.get('fliplr') else None,
            'augment_hsv' : RandAugment.augment_hsv if augment_params.get('augment_hsv') else None,
            'hist_equalize' : RandAugment.hist_equalize if augment_params.get('hist_equalize') else None,
            'solarize' : RandAugment.solarize if augment_params.get('solarize') else None,
            'posterize': RandAugment.posterize if augment_params.get('posterize') else None,
            'adjust_brightness': RandAugment.adjust_brightness if augment_params.get('adjust_brightness') else None,
            'invert' : RandAugment.invert if augment_params.get('invert') else None,
            'contrast': RandAugment.contrast if augment_params.get('contrast') else None,
            'shearX' : RandAugment.shear_x if augment_params.get('shearX') else None,
            'shearY' : RandAugment.shear_y if augment_params.get('shearY') else None,
            'translateX' : RandAugment.translate_x if augment_params.get('translateX') else None,
            'translateY' : RandAugment.translate_y if augment_params.get('translateY') else None,
            'sharpness' : RandAugment.sharpness if augment_params.get('sharpness') else None,
            'cutout' : RandAugment.cutout if augment_params.get('cutout') else None,
            'rotate' : RandAugment.rotate if augment_params.get('rotate') else None,
            'cut_25_left' : RandAugment.cut_25_left if augment_params.get('cut_25_left') else None,
            'cut_25_right': RandAugment.cut_25_right if augment_params.get('cut_25_right') else None,
            'cut_25_above': RandAugment.cut_25_above if augment_params.get('cut_25_above') else None,
            'cut_25_under': RandAugment.cut_25_under if augment_params.get('cut_25_under') else None,
            'blur_img': RandAugment.blur_img if augment_params.get('blur_img') else None,
            'center_crop': RandAugment.center_crop if augment_params.get('center_crop') else None,
            'augment_RGB': RandAugment.augment_RGB if augment_params.get('augment_RGB') else None,
            # 'random_crop':random_crop
        }
        self.ARGS_LIMIT = {
            'fliplr' : augment_params.get('fliplr'),
            'augment_hsv': augment_params.get('augment_hsv'),
            'hist_equalize' : augment_params.get('hist_equalize'),
            'solarize' : augment_params.get('solarize'),
            'posterize': augment_params.get('posterize'),
            'adjust_brightness': augment_params.get('adjust_brightness'),
            'invert' : augment_params.get('invert'),
            'contrast': augment_params.get('contrast'),
            'shearX' : augment_params.get('shearX'),
            'shearY' : augment_params.get('shearY'),
            'translateX' : augment_params.get('translateX'),
            'translateY' : augment_params.get('translateY'),
            'sharpness' : augment_params.get('sharpness'),
            'cutout' : augment_params.get('cutout'),
            'rotate' :  augment_params.get('rotate'),
            'cut_25_left' : augment_params.get('cut_25_left'),
            'cut_25_right': augment_params.get('cut_25_right'),
            'cut_25_above': augment_params.get('cut_25_above'),
            'cut_25_under': augment_params.get('cut_25_under'),
            'blur_img': augment_params.get('blur_img'),
            'center_crop': augment_params.get('center_crop'),
            'augment_RGB': augment_params.get('augment_RGB')
        }
        self.policy = list(k for k,v in self.AUGMENT_FUNCTION.items() if v)
        # print(self.policy)
    def mixup(img1,img2,factor):
        img = img1.astype('float')* factor + img2.astype('float') * (1-factor)
        img = np.clip(img, 0,255)
        img = img.astype('uint8')
        return img

    def augment_fliplr(img,level):
        if random.random() < level:
            return np.fliplr(img)
        return img 

    def augment_hsv(im, level=None, hgain=0.000, sgain=0.0005, vgain=0.005):
        im = im.copy()
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            r = np.array([hgain,sgain,vgain]) + 1
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
        return im_hsv

    def hist_equalize(im, level=None,clahe=True, bgr=True):
        im = im.copy()
        # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
        yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
        if clahe:
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

    def solarize(image, level=128):
        threshold = level
        image = image.copy()
        # For each pixel in the image, select the pixel
        # if the value is less than the threshold.
        # Otherwise, subtract 255 from the pixel.
        return np.where(image <= threshold, image, 255 - image)

    def posterize(img, level=3):
        bits = level
        shift = 8 - bits
        # img = img >> shift
        img = np.left_shift(img,shift)
        img = np.right_shift(img,shift)
        return img.astype('uint8')

    def adjust_brightness(img,level=0.5):
        factor = level
        degenerate = np.zeros(img.shape,dtype='uint8')
        img = RandAugment.mixup(img,degenerate,factor)
        return img

    def invert(img,level=None):
        return 255-img

    def contrast(img,factor=0.5): 
        degenerate = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return RandAugment.mixup(img,degenerate,factor)

    def shear_x(img,level=0.4,mode='reflect'):
        M = np.array([[1, level, 0],
                    [0,  1   , 0],
                    [0,  0   , 1]],dtype='float')
        height,width,_ = img.shape
        option_mode ={
            'reflect'  : cv2.BORDER_REPLICATE,
            'constant' : cv2.BORDER_CONSTANT
        }
        mode = option_mode[mode]
        sheared_img = cv2.warpPerspective(img, M, (width, height), borderMode=mode)
        return sheared_img

    def shear_y(img,level=0.4,mode='reflect'):
        M = np.array([[1,      0   , 0],
                    [level,  1   , 0],
                    [0,      0   , 1]],dtype='float')
        height,width,_ = img.shape
        option_mode ={
            'reflect'  : cv2.BORDER_REPLICATE,
            'constant' : cv2.BORDER_CONSTANT
        }
        mode = option_mode[mode]
        sheared_img = cv2.warpPerspective(img, M, (width, height), borderMode=mode)
        return sheared_img

    def translate_x(img,level,mode='reflect'): 
        height,width,_ = img.shape
        option_mode ={
            'reflect'  : cv2.BORDER_REPLICATE,
            'constant' : cv2.BORDER_CONSTANT
        }
        mode = option_mode[mode]
        translate_pixel = int(width * level)
        M = np.array([[1,      0   , translate_pixel],
                    [level,  1   , 0],
                    [0,      0   , 1]],dtype='float')
        translate_img = cv2.warpPerspective(img, M, (width, height), borderMode=mode)
        return translate_img

    def translate_y(img,level,mode='reflect'): 
        height,width,_ = img.shape
        option_mode ={
            'reflect'  : cv2.BORDER_REPLICATE,
            'constant' : cv2.BORDER_CONSTANT
        }
        mode = option_mode[mode]
        translate_pixel = int(width * level)
        M = np.array([[1,      0   , 0],
                    [level,  1   , translate_pixel],
                    [0,      0   , 1]],dtype='float')
        translate_img = cv2.warpPerspective(img, M, (width, height), borderMode=mode)
        return translate_img

    # def sharpness(img,): 
    #     kernel = np.array(
    #     [[1, 1, 1], 
    #     [1, 5, 1], 
    #     [1, 1, 1]], dtype=tf.float32,
    #     shape=[3, 3, 1, 1]) / 13.
    #     cv2.

    def cutout(img,level,**kwargs): 
        img = img.copy()
        height,width ,_ = img.shape 
        padding_size = int(height*level),int(width*level)
        value = kwargs.get('value') 
        cordinate_h = np.random.randint(0,height-padding_size[0])
        cordinate_w = np.random.randint(0,width-padding_size[1])
        img[cordinate_h:cordinate_h+padding_size[0],cordinate_w:cordinate_w+padding_size[1],:] = 255
        return img 

    def rotate(image, level=45, center = None, scale = 1.0):
        angle=level
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h),borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def cut_25_under(img,level=0.25):
        ratio = level
        height,width,_ = img.shape
        new_height = int((1-ratio)*height)
        img_ = img[:new_height,:,:]
        
        height,width,_ = img_.shape 
        if height > width :
                    img2 = np.pad(img_,[[0,0],[(height-width)//2,(height-width)//2],[0,0]],mode='constant',constant_values=255)
        else:
                    img2 = np.pad(img_,[[(width-height)//2,(width-height)//2],[0,0],[0,0]],mode='constant',constant_values=255)
        img2 = cv2.resize(img2,(224,224))
        return img2

    def cut_25_above(img,level=0.25):
        ratio = level
        height,width,_ = img.shape
        new_height = int(ratio*height)
        img_ = img[new_height:,:,:]
        
        height,width,_ = img_.shape 
        if height > width :
                    img2 = np.pad(img_,[[0,0],[(height-width)//2,(height-width)//2],[0,0]],mode='constant',constant_values=255)
        else:
                    img2 = np.pad(img_,[[(width-height)//2,(width-height)//2],[0,0],[0,0]],mode='constant',constant_values=255)
        img2 = cv2.resize(img2,(224,224))
        return img2

    def cut_25_right(img,level=0.25):
        ratio = level
        height,width,_ = img.shape
        new_width = int((1-ratio)*width)
        img_ = img[:,:new_width,:]
        height,width,_ = img_.shape 
        if height > width :
                    img2 = np.pad(img_,[[0,0],[(height-width)//2,(height-width)//2],[0,0]],mode='constant',constant_values=255)
        else:
                    img2 = np.pad(img_,[[(width-height)//2,(width-height)//2],[0,0],[0,0]],mode='constant',constant_values=255)    
        img2 = cv2.resize(img2,(224,224))
        return img2

    def cut_25_left(img,level=0.25):
        ratio = level
        height,width,_ = img.shape
        new_width = int(ratio*width)
        img_ = img[:,new_width:,:]
        height,width,_ = img_.shape 
        if height > width :
                    img2 = np.pad(img_,[[0,0],[(height-width)//2,(height-width)//2],[0,0]],mode='constant',constant_values=255)
        else:
                    img2 = np.pad(img_,[[(width-height)//2,(width-height)//2],[0,0],[0,0]],mode='constant',constant_values=255)    
        img2 = cv2.resize(img2,(224,224))
        return img2    

    def blur_img(img,level=10):
        level = int(level)
        ksize = (level,level)
        image = cv2.blur(img, ksize) 
        return image 

    def center_crop(img,level=0.15):
        height,width,_ = img.shape
        
        img = img[int(level*height):int((1-level)*height), int(level*width):int((1-level)*width)]
        img = cv2.resize(img,(width,height))
        return img

    def augment_RGB(img, level=0.5):
        R_gain,G_gain,B_gain = np.random.uniform(-1, 1, 3) * [level, level, level]
        B,G,R = cv2.split(img)
        B = B.astype('float')*(1+B_gain)
        G = G.astype('float')*(1+G_gain)
        R = R.astype('float')*(1+R_gain)
        
        B = np.clip(B,0,255).astype('uint8')
        G = np.clip(G,0,255).astype('uint8')
        R = np.clip(R,0,255).astype('uint8')

        image = cv2.merge((B,G,R))
        return image
    def __call__(self,img):
        augmenters = random.choices(self.policy, k=self.num_layers)
        # print(augmenters)
        for augmenter in augmenters:
            level = random.random()
            # try:
            min_arg,max_arg = self.ARGS_LIMIT[augmenter]

            level = min_arg + (max_arg - min_arg) * level
            img = self.AUGMENT_FUNCTION[augmenter](img,level=level)
            # except:
                # print(augmenter)
        return img 

def augmentation_test():
    img_org = cv2.imread('test.jpg')
    import yaml
    augment_params = yaml.safe_load(open('config/default/train_config.yaml')).get('augment_params')
    augmenter = RandAugment(augment_params=augment_params)#(num_layers=1)
    for _ in range(10000):
        img_aug = augmenter(img_org)
        img_pad = preprocess(img_aug,224)
        # cv2.imshow('a',img_org)
        # cv2.imshow('b',img_aug)
        # cv2.imshow('c',img_pad)
        # if cv2.waitKey(0)==ord('q'):
            # exit()

class HumanColorAugment():
    """
    augmentation for human color project
    """
    def __init__(self,**kwargs):
        pass
    
    def multiply(self,img,labels):
        channel = list(cv2.split(img))
        a = random.random() * 0.1 + 0.95
        b = random.random() * 0.1 + 0.95
        c = random.random() * 0.1 + 0.95
        channel[0] = channel[0].astype('float32') * a
        channel[0] = np.clip(channel[0],0,255).astype('uint8')
        labels[2] = max(min(labels[2]*a,255),0)
        labels[5] = max(min(labels[5]*a,255),0)

        channel[1] = channel[1].astype('float32') * b
        channel[1] = np.clip(channel[1],0,255).astype('uint8')
        labels[1] = max(min(labels[1]*b,255),0)
        labels[4] = max(min(labels[4]*b,255),0)

        channel[2] = channel[2].astype('float32') * c
        channel[2] = np.clip(channel[2],0,255).astype('uint8')
        labels[0] = max(min(labels[0]*c,255),0)
        labels[3] = max(min(labels[3]*c,255),0)

        temp = []
        temp.append([channel[0],labels[2],labels[5]])
        temp.append([channel[1],labels[1],labels[4]])
        temp.append([channel[2],labels[0],labels[3]])
        random.shuffle(temp)
        img = cv2.merge([temp[0][0],temp[1][0],temp[2][0]])
        labels = []
        labels.append(temp[2][1])
        labels.append(temp[1][1])
        labels.append(temp[0][1])
        labels.append(temp[2][2])
        labels.append(temp[1][2])
        labels.append(temp[0][2])

        return img,labels

    def add(self,img,labels):
        channel = list(cv2.split(img))
        a = random.random() * 10 - 5
        b = random.random() * 10 - 5
        c = random.random() * 10 - 5
        channel[0] = channel[0].astype('float32') + a
        channel[0] = np.clip(channel[0],0,255).astype('uint8')
        labels[2] = max(min(labels[2]+a,255),0)
        labels[5] = max(min(labels[5]+a,255),0)

        channel[1] = channel[1].astype('float32') + b
        channel[1] = np.clip(channel[1],0,255).astype('uint8')
        labels[1] = max(min(labels[1]+b,255),0)
        labels[4] = max(min(labels[4]+b,255),0)

        channel[2] = channel[2].astype('float32') + c
        channel[2] = np.clip(channel[2],0,255).astype('uint8')
        labels[0] = max(min(labels[0]+c,255),0)
        labels[3] = max(min(labels[3]+c,255),0)

        temp = []
        temp.append([channel[0],labels[2],labels[5]])
        temp.append([channel[1],labels[1],labels[4]])
        temp.append([channel[2],labels[0],labels[3]])
        random.shuffle(temp)
        img = cv2.merge([temp[0][0],temp[1][0],temp[2][0]])
        labels = []
        labels.append(temp[2][1])
        labels.append(temp[1][1])
        labels.append(temp[0][1])
        labels.append(temp[2][2])
        labels.append(temp[1][2])
        labels.append(temp[0][2])

        return img,labels

    def __call__(self,img,labels):
        if random.random() > 0.5:
            img,labels = self.multiply(img,labels)    
        else:
            img,labels = self.add(img,labels)
        if random.random() > 0.5:
            img = np.fliplr(img)
        return img,labels



























if __name__ =='__main__':
    augmentation_test()
