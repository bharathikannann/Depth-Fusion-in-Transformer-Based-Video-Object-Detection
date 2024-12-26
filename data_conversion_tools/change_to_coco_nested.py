# Files used to convert the Yolo txt files to the COCO format JSON files
# Supports nested folders for videos

import os
import json
import cv2
import shutil
import argparse
from tqdm import tqdm
import glob
from PIL import Image

def get_annotations(images_dir, labels_dir, output_dir, annotations_file_name, categories_file = None):
    
    '''
    directory - image directory
    output_dir - Directory to store annotations
    unlabelled_images - list of unlabelled images
    annotation_file_name - name of the annotation file name to store in the output_dir
    categories_file - list of categories 
    '''

    if categories_file:
        with open(categories_file,'r') as f1:
            lines1 = f1.readlines()
    else:
        lines1 = ['hand']
        
    # create the template
    categories = []
    for j,label in enumerate(lines1):
        label = label.strip()
        categories.append({'id':j+1,'name':label,'supercategory': label})

    write_json_data = dict()
    write_json_data['info'] = {'description': None, 'url': None, 'version': None, 'year': 2022, 'contributor': None, 'date_created': None}
    write_json_data['licenses'] = [{'id': 1, 'name': None, 'url': None}]
    write_json_data['categories'] = categories
    write_json_data['images'] = []
    write_json_data['annotations'] = [] 
    write_json_data['videos'] = []

    directory = os.path.join(images_dir,"")
    labels_dir = os.path.join(labels_dir,"")

    image_id = 1
    bbox_id = 1

    video_id = 0
    
    # walk through all image folders
    for dir_no, first_image_folder in tqdm(enumerate(os.listdir(directory)), desc = "Processing folders"):
        temp_x = first_image_folder
        if first_image_folder not in ["ifm_2019"]:
            continue
        tqdm.write(f"\nProcessing folder: {first_image_folder}")
        out_dir_name = first_image_folder
        first_label_folder = os.path.join(labels_dir, first_image_folder, "labels/yolov3")
        first_image_folder = os.path.join(images_dir, first_image_folder, "images")
        for video_no, image_folder in enumerate(os.listdir(first_image_folder)):
            tqdm.write(f"\nProcessing video: {image_folder}")
            directory_labels = os.fsencode(os.path.join(first_label_folder,image_folder))
            directory_images = os.fsencode(os.path.join(first_image_folder,image_folder))
            
            if "pick" not in image_folder:
                video_id+=1
                write_json_data['videos'].append({'id':video_id,'file_name':os.fsdecode(directory_images)})
            
                # sort the images and labels
                if isinstance(directory_images, bytes):
                    directory_images = directory_images.decode()

                image_files = [os.path.relpath(x, directory_images) for x in glob.glob(os.path.join(directory_images, '**', '*'), recursive=True)]
                # image_files = [os.path.basename(x) for x in glob.glob(os.path.join(directory_images, '*'))]
                # if first_image_folder in ["ifm_2019"]:
                #     sorted_image_files = sorted(image_files, key=lambda x: os.path.splitext(os.path.basename(x))[0])
                # else:
                sorted_image_files = sorted(image_files, key=lambda x: os.path.splitext(os.path.basename(x))[0])

                for frame_id, file in tqdm(enumerate(sorted_image_files), desc="Processing images"): # walk through all files in the folder
                    # if frame_id > 100:
                    #     break
                    filename = os.fsdecode(file)
                    if filename.endswith(".jpg") or filename.endswith('jpeg') or filename.endswith('png'): # choose only the images
                        img_path = (os.path.join(directory_images, filename))
                        img_name = os.path.relpath(img_path, directory_images)
                        file_name_without_ext = os.path.splitext(img_name)[0]
                        yolo_annotation_path  = os.path.join(directory_labels.decode("utf-8"), file_name_without_ext+ "." + 'txt') # get annotations for the image
                        img_data = {} 
                        height,width = cv2.imread(img_path).shape[:2]
                        img_data['file_name'] = os.path.join(os.path.join(out_dir_name, "images", image_folder), img_name)
                        img_data['height'] = height
                        img_data['width'] = width
                        img_data['video_id'] = video_id
                        img_data['frame_id'] = frame_id
                        img_data['date_captured'] = ''
                        img_data['id'] = image_id 
                        img_data['license'] = 1
                        write_json_data['images'].append(img_data)
                        try:
                            with open(yolo_annotation_path,'r') as f2:
                                annotation_file_lines = f2.readlines()
                        except:
                            image_id = image_id+1
                            continue
                        
                        if len(annotation_file_lines) == 0:
                            image_id = image_id+1
                            continue
                        
                        for i,line in enumerate(annotation_file_lines): # loop through all annotations
                            line = line.split(' ')
                            annotation_data = {}
                            class_id, x_yolo, y_yolo, width_yolo, height_yolo = line[0:]
                            class_id, x_yolo, y_yolo, width_yolo, height_yolo = int(class_id), float(x_yolo), float(y_yolo), float(width_yolo), float(height_yolo)
                            annotation_data['id'] = bbox_id
                            annotation_data['image_id'] = image_id
                            annotation_data['video_id'] = dir_no+1
                            annotation_data['category_id'] = class_id+1
                            annotation_data['iscrowd'] = False
                            annotation_data['occluded'] = False
                            annotation_data['generated'] = False
                            h,w = abs(height_yolo*height),abs(width_yolo*width)
                            annotation_data['area']  = h * w
                            x_coco = round(x_yolo*width -(w/2))
                            y_coco = round(y_yolo*height -(h/2))
                            if x_coco <0: # check if x_coco extends out of the image boundaries
                                w = w + x_coco
                                x_coco = 0
                            if y_coco <0: # check if y_coco extends out of the image boundaries
                                h = h + y_coco
                                y_coco = 0
                            if x_coco+w > width:
                                w = width - x_coco
                            if y_coco+h > height:
                                h = height - y_coco
                            
                            annotation_data['bbox'] = [x_coco,y_coco,w,h]
                            annotation_data['segmentation'] = []
                            # annotation_data['segmentation'] = [[x_coco,y_coco,x_coco+w,y_coco, x_coco+w, y_coco+h, x_coco, y_coco+h]]
                            write_json_data['annotations'].append(annotation_data)
                            bbox_id+=1

                        image_id = image_id+1
                        continue
                    else:
                        continue
            else:
                for sub_video_no, sub_image_folder in enumerate(os.listdir(directory_images)):
                    tqdm.write(f"\nProcessing sub video: {sub_image_folder}")
                    sub_directory_labels = os.fsencode(os.path.join(directory_labels,sub_image_folder))
                    sub_directory_images = os.fsencode(os.path.join(directory_images,sub_image_folder))
                    
                    video_id+=1
                    write_json_data['videos'].append({'id':video_id,'file_name':os.fsdecode(sub_directory_images)})
            
                    # sort the images and labels
                    if isinstance(sub_directory_images, bytes):
                        sub_directory_images = sub_directory_images.decode()

                    sub_image_files = [os.path.relpath(x, sub_directory_images) for x in glob.glob(os.path.join(sub_directory_images, '**', '*'), recursive=True)]
                    # image_files = [os.path.basename(x) for x in glob.glob(os.path.join(directory_images, '*'))]
                    # if first_image_folder in ["ifm_2019"]:
                    #     sorted_image_files = sorted(image_files, key=lambda x: os.path.splitext(os.path.basename(x))[0])
                    # else:
                    sorted_image_files = sorted(sub_image_files, key=lambda x: os.path.splitext(os.path.basename(x))[0])

                    for frame_id, file in tqdm(enumerate(sorted_image_files), desc="Processing images"): # walk through all files in the folder
                        # if frame_id > 100:
                        #     break
                        filename = os.fsdecode(file)
                        if filename.endswith(".jpg") or filename.endswith('jpeg') or filename.endswith('png'): # choose only the images
                            img_path = (os.path.join(sub_directory_images, filename))
                            img_name = os.path.relpath(img_path, sub_directory_images)
                            file_name_without_ext = os.path.splitext(img_name)[0]
                            yolo_annotation_path  = os.path.join(sub_directory_labels.decode("utf-8"), file_name_without_ext+ "." + 'txt') # get annotations for the image
                            img_data = {} 
                            height,width = cv2.imread(img_path).shape[:2]
                            img_data['file_name'] = os.path.join(os.path.join(out_dir_name, "images", image_folder, os.fsdecode(sub_image_folder)), img_name)
                            img_data['height'] = height
                            img_data['width'] = width
                            img_data['video_id'] = video_id
                            img_data['frame_id'] = frame_id
                            img_data['date_captured'] = ''
                            img_data['id'] = image_id 
                            img_data['license'] = 1
                            write_json_data['images'].append(img_data)
                            try:
                                with open(yolo_annotation_path,'r') as f2:
                                    annotation_file_lines = f2.readlines()
                            except:
                                image_id = image_id+1
                                continue
                            
                            if len(annotation_file_lines) == 0:
                                image_id = image_id+1
                                continue
                            
                            for i,line in enumerate(annotation_file_lines): # loop through all annotations
                                line = line.split(' ')
                                annotation_data = {}
                                class_id, x_yolo, y_yolo, width_yolo, height_yolo = line[0:]
                                class_id, x_yolo, y_yolo, width_yolo, height_yolo = int(class_id), float(x_yolo), float(y_yolo), float(width_yolo), float(height_yolo)
                                annotation_data['id'] = bbox_id
                                annotation_data['image_id'] = image_id
                                annotation_data['video_id'] = dir_no+1
                                annotation_data['category_id'] = class_id+1
                                annotation_data['iscrowd'] = False
                                annotation_data['occluded'] = False
                                annotation_data['generated'] = False
                                h,w = abs(height_yolo*height),abs(width_yolo*width)
                                annotation_data['area']  = h * w
                                x_coco = round(x_yolo*width -(w/2))
                                y_coco = round(y_yolo*height -(h/2))
                                if x_coco <0: # check if x_coco extends out of the image boundaries
                                    w = w + x_coco
                                    x_coco = 0
                                if y_coco <0: # check if y_coco extends out of the image boundaries
                                    h = h + y_coco
                                    y_coco = 0
                                if x_coco+w > width:
                                    w = width - x_coco
                                if y_coco+h > height:
                                    h = height - y_coco
                                
                                annotation_data['bbox'] = [x_coco,y_coco,w,h]
                                annotation_data['segmentation'] = []
                                # annotation_data['segmentation'] = [[x_coco,y_coco,x_coco+w,y_coco, x_coco+w, y_coco+h, x_coco, y_coco+h]]
                                write_json_data['annotations'].append(annotation_data)
                                bbox_id+=1

                            image_id = image_id+1
                            continue
                        else:
                            continue
            
    new_dir = os.path.join(output_dir, "")
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    coco_format_save_dir = new_dir
    
    coco_format_save_path = os.path.join(coco_format_save_dir, annotations_file_name + '.json')
    with open(coco_format_save_path,'w') as fw:
        json.dump(write_json_data,fw) 
    
    return write_json_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--labels_dir', type = str, required = False)
    parser.add_argument('--output_dir', type=str, default = "./", required = False)
    args = parser.parse_args()

    # train_dir = args.train_dir
    images_dir = args.images_dir
    labels_dir = args.labels_dir

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok = True)
        output_dir = args.output_dir
    else:
        output_dir = args.images_dir
    
    _ = get_annotations(images_dir, labels_dir, output_dir, annotations_file_name = "train")

    print("Converted the dataset to COCO format")