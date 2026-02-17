# -*- coding: cp1251 -*-
import xml.etree.ElementTree as ET
import streamlit as st
import numpy as np

import zipfile
import json
import math
import io


@st.cache_data(show_spinner = False, max_entries = 5)
def ExportToCVAT(imageData, BLOBs): 
    manifest_jsonl_file = f'''{{"version":"1.1"}}\n{{"type":"images"}}\n{{"name":"{imageData['name']}","extension":".tif","width":{imageData['width']},"height":{imageData['height']},"meta":{{"related_images":[]}}}}'''
    
    # not changed
    task_json_file = """
        {
            "name":"Nano labeling",
            "bug_tracker":"",
            "status":"annotation",
            "subset":"",
            "labels":[{
                "name":"Nanoparticle",
                "color":"#ff355e",
                "attributes":[],
                "type":"polyline",
                "sublabels":[]
            }],
            "version":"1.0",
            "data":{
                "chunk_size":56,
                "image_quality":100,
                "start_frame":0,
                "stop_frame":0,
                "storage_method":"cache",
                "storage":"local",
                "sorting_method":"lexicographical",
                "chunk_type":"imageset",
                "deleted_frames":[]
            },
            "jobs":[{
                "status":"annotation",
                "type":"annotation",
                "start_frame":0,
                "stop_frame":0
            }]
        }
    """ 

    shapes = ""
    for i, blob in enumerate(BLOBs):
        y, x, d = blob

        shape_element = f"""{{
            "type":"polyline",
            "occluded":false,
            "outside":false,
            "z_order":0,
            "rotation":0.0,
            "points":[{x},{y - d/2},{x},{y + d/2}],
            "frame":0,
            "group":0,
            "source":"manual",
            "attributes":[],
            "elements":[],
            "label":"Nanoparticle"
        }}""" 

        shapes += shape_element
        if i < (len(BLOBs) - 1):
            shapes += ",\n"

    annotations_json_file = f"""[{{
        "version":0,
        "tags":[],
        "shapes":[{shapes}],
        "tracks":[]
    }}]"""


    files = {
        f"data/{imageData['name']}.tif": imageData['buffer'],
        'data/manifest.jsonl': manifest_jsonl_file,
        'annotations.json': annotations_json_file,
        'task.json': task_json_file
    }
    
    zipBuffer = io.BytesIO()
    with zipfile.ZipFile(zipBuffer, 'w') as tempZipFile:
        for file_path, content in files.items():
            if isinstance(content, str):
                content = content.encode('utf-8')
            tempZipFile.writestr(file_path, content)
    
    zipBuffer.seek(0)
    return zipBuffer


@st.cache_data(show_spinner = False, max_entries = 5)
def ImportJobFromCVAT(jobCVAT):
    with zipfile.ZipFile(jobCVAT, 'r') as tempZipFile:
        annotations = tempZipFile.read('annotations.xml')
        annotations = annotations.decode('utf-8')

    BLOBs = []

    root = ET.fromstring(annotations)
    for image in root.findall('image'):
        imgName = image.get('name')        
        imgWidth = int(image.get('width'))
        imgHeight = int(image.get('height'))
        for polyline in image.findall('polyline'):
            points = polyline.get('points')
        
            pairs = points.split(";")
            coordinates = [tuple(map(float, pair.split(","))) for pair in pairs]

            d = math.dist(coordinates[0], coordinates[1])
            x = (coordinates[0][0] + coordinates[1][0]) / 2
            y = (coordinates[0][1] + coordinates[1][1]) / 2

            BLOBs.append([y, x, d])

    return np.array(BLOBs), imgName, [imgWidth, imgHeight]


@st.cache_data(show_spinner = False, max_entries = 5)
# taskCVAT: path to zip file CVAT with labeled particles
def ImportTaskFromCVAT(taskCVAT):
    with zipfile.ZipFile(taskCVAT, 'r') as tempZipFile:
        annotations = tempZipFile.read('annotations.json')
        annotations = annotations.decode('utf-8')

        manifest = tempZipFile.read('data/manifest.jsonl')
        manifest = manifest.decode('utf-8')
        temp = json.loads(manifest.split('\n')[-1])
        imgFileName = temp['name'] + temp['extension']        
        imageBytes = tempZipFile.read(f'data/{imgFileName}')
    
    BLOBs = []

    annotations = json.loads(annotations)
    for shape in annotations[0]['shapes']:
        points = shape['points']

        coordinates = [[points[0], points[1]], [points[2], points[3]]]

        d = math.dist(coordinates[0], coordinates[1])
        x = (coordinates[0][0] + coordinates[1][0]) / 2
        y = (coordinates[0][1] + coordinates[1][1]) / 2
        
        BLOBs.append([y, x, d]) #! particle diameters !

    return np.array(BLOBs), imgFileName, io.BytesIO(imageBytes)


if __name__ == "__main__": 
    
    from PIL import Image
    taskFile = r"D:\Загрузки\task_nano labeling_backup_2025_09_11_10_57_41.zip"

    temp = ImportTaskFromCVAT(taskFile)
    print(1)

    # jobFile = r"D:\Загрузки\job_2955897_annotations_2025_09_09_13_06_06_cvat for images 1.1.zip"
    
    # BLOBs, imgName, size = ImportJobFromCVAT(jobFile)
    # print(size, imgName, BLOBs)

        
    # img = Image.open(r"C:\Users\Muwa\Desktop\data\142-S3-A17-100k-disordered.tif").convert('L')    
    # img_buffer = io.BytesIO()
    # img.save(img_buffer, format = 'TIFF')

    # imageData= {
    #     'name': "142-S3-A17-100k-disordered",
    #     'width': 1280,
    #     'height': 1024,
    #     'buffer': img_buffer.getvalue()
    # }

    # BLOBs = [
    #     [10, 20, 5],
    #     [20, 40, 7],
    #     [25, 25, 10]
    #     ]

    # zip_buffer = ExportToCVAT(imageData, BLOBs)    

    # with open(f"backup-{time.strftime('%Y-%m-%d-%H-%M-%S')}.zip", 'wb') as f:
    #     f.write(zip_buffer.getvalue())