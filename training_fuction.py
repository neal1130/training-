import cv2
import numpy as np
from tqdm import tqdm

def flip(raw_list, mask_list, direction):
    flip_raw = [cv2.flip(img, direction) for img in tqdm(raw_list)]
    flip_mask = [cv2.flip(img, direction) for img in tqdm(mask_list)]

    return flip_raw, flip_mask

def rotate(raw_list, mask_list, degree):
    (h,w) = raw_list[0].shape
    M = cv2.getRotationMatrix2D((h/2,w/2), degree, 1)
    rotated_data = [cv2.warpAffine(img, M, (w, h)) for img in tqdm(raw_list)]
    rotated_mask = [cv2.warpAffine(img, M, (w, h)) for img in tqdm(mask_list)]

    return rotated_data, rotated_mask

def bright_multiply(raw_list, mask_list, multiply_number, thershold):
    brighten_data = [cv2.GaussianBlur(img,(3,3),0) for img in tqdm(raw_list)]

    for idx in tqdm(range(len(raw_list))):
        data = brighten_data[idx].astype('float64')  # Cast to float64 before the operation
        mask = data > thershold
        data[mask] *= multiply_number
        brighten_data[idx] = data.astype('uint16')  # Optional: Cast back to uint16 if needed

    return brighten_data, mask_list

def bright_multiply_add(raw_list, mask_list, multiply_number,add_number, thershold):
    brighten_data = [cv2.GaussianBlur(img,(3,3),0) for img in tqdm(raw_list)]

    for idx in tqdm(range(len(raw_list))):
        data = brighten_data[idx].astype('float64')  # Cast to float64 before the operation
        mask = data > thershold
        data[mask] *= multiply_number
        data[mask]=data[mask]+add_number
        brighten_data[idx] = data.astype('uint16')  # Optional: Cast back to uint16 if needed

    return brighten_data, mask_list

def bright_gamma(raw_list, mask_list, gamma_number):
    brighten_data = []
    for img in tqdm(raw_list):
        img_max=img.max()
        img_min=img.min()
        nor_img=(img-img_min)/(img_max-img_min)
        result=nor_img**gamma_number
        brighten_data.append(result)
    return brighten_data, mask_list

def gammahigh(raw_list, mask_list, gamma_number):
    
     
     brighten_data = []
     for img in tqdm(raw_list):
         img=np.where(img>800,0,img)
         img=np.where(img>220,220,img)
         img_max=img.max()
         img_min=img.min()
         nor_img=(img-img_min)/(img_max-img_min)
         result=nor_img**gamma_number
         brighten_data.append(result)
     return brighten_data, mask_list
    

    

def bright_add(raw_list, mask_list, multiply_number, thershold):
    brighten_data = [cv2.GaussianBlur(img,(3,3),0) for img in tqdm(raw_list)]

    for idx in tqdm(range(len(raw_list))):
        data = brighten_data[idx]
        mask = data > thershold
        data[mask] = data[mask] + multiply_number
      #  _, data = cv2.threshold(data, 15000, 65535, cv2.THRESH_TRUNC)
        brighten_data[idx] = data

    return brighten_data, mask_list

def adjust_gamma(raw_list, mask_lsit, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected_image = [cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) for img in tqdm(raw_list)]
    corrected_image = [cv2.LUT(img, table) for img in tqdm(corrected_image)]
    corrected_image = [cv2.normalize(img, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)  for img in tqdm(corrected_image)]

    return corrected_image, mask_lsit

