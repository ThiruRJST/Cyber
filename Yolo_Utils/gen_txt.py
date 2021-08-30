from numpy.core.defchararray import less
import pandas as pd
import glob
import numpy as np
import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

data = pd.read_csv("/home/lustbeast/kaggle/CORONA/csv/train_image_level.csv")
data['label'] = data.label.apply(lambda x:x.split(' ')[0])
data['id'] = data.id.apply(lambda y:y.replace('_image','.dcm'))
uniq_ids = data.StudyInstanceUID.unique()
image_ids = data.id.values

print(f'Unique_StudyIDS:{len(uniq_ids)}, Image IDS:{len(image_ids)}')

dataset_root = "/media/lustbeast/6A38216B38213789/kaggle/siim-covid19-detection/train"

datapaths = []
#print(datapaths)

def gen_paths(df):
    for i in range(len(df)):
        try:
            p = glob.glob(f"{dataset_root}/{data.loc[i,'StudyInstanceUID']}/*/{data.loc[i,'id']}")[0]
            datapaths.append(p)
        except IndexError as e:
            print(e)
        
    return datapaths

def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im

def gen_helper(df,fold,subset,dest_folder):

    for i in tqdm(range(len(df))):
            dcm = read_xray(os.path.join("/media/lustbeast/6A38216B38213789/kaggle/siim-covid19-detection/train",df.loc[i,'StudyInstanceUID'],df.loc[i,'Series'])+f"/{df.loc[i,'id']}")
            dcm = resize(dcm,size=256)
            assert dest_folder is not None
            dcm.save(os.path.join(dest_folder,subset,'images',str(fold),df.loc[i,'id'].replace('.dcm','.jpg')))
            #f = open(os.path.join(dest_folder,subset,'labels',str(fold),df.loc[i,'id'].replace('.dcm','.txt')),'w')
            if df.loc[i,'label'] == 'opacity':
                if len(df.loc[i,'x'].split(',')) == 2:
                    x_obj1,x_obj2 = df.loc[i,'x'].split(',')[0],df.loc[i,'x'].split(',')[1]
                    y_obj1,y_obj2 = df.loc[i,'y'].split(',')[0],df.loc[i,'y'].split(',')[1]
                    width_obj1,width_obj2 = df.loc[i,'width'].split(',')[0],df.loc[i,'width'].split(',')[1]
                    height_obj1,height_obj2 = df.loc[i,'height'].split(',')[0],df.loc[i,'height'].split(',')[1]

                    with open(os.path.join(dest_folder,subset,'labels',str(fold),df.loc[i,'id'].replace('.dcm','.txt')),'w') as f:
                        f.write(f"0 {x_obj1} {y_obj1} {width_obj1} {height_obj1}\n")
                        #f.write("\n")
                        f.write(f"0 {x_obj2} {y_obj2} {width_obj2} {height_obj2}")
                        f.close()

                elif len(df.loc[i,'x'].split(',')) == 1:
                    with open(os.path.join(dest_folder,subset,'labels',str(fold),df.loc[i,'id'].replace('.dcm','.txt')),'w') as f:
                        x,y,width,height2 = df.loc[i,'x'],df.loc[i,'y'],df.loc[i,'width'],df.loc[i,'height']
                        f.write(f"0 {x} {y} {width} {height2}")
                        f.close()
            
            elif df.loc[i,'label'] == 'none':
                with open(os.path.join(dest_folder,subset,'labels',str(fold),df.loc[i,'id'].replace('.dcm','.txt')),'w') as f:
                    f.write(f"")
                    f.close()


def txt_gen(df,splits=['Train','Val'],num_folds=5,dest_folder="/home/lustbeast/kaggle/CORONA/Data"):
    for s in splits:
        if s == 'Train':
            for n in range(num_folds):
                train_df = df[df['folds'] != n]
                train_df = train_df.reset_index()
                gen_helper(df=train_df,fold=n,subset=s,dest_folder=dest_folder)
        elif s == 'Val':
            for n in range(num_folds):
                val_df = df[df['folds'] == n]
                val_df = val_df.reset_index()
                gen_helper(df=val_df,fold=n,subset=s,dest_folder=dest_folder)




