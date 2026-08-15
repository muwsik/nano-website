import xml.etree.ElementTree as ET
import numpy as np

import zipfile
import json
import math
import io


def exportToCVAT(imageData, x, y, diameter): 
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

    shapes = []
    for _x, _y, _d in zip(x, y, diameter):
        shapes.append(f"""{{
            "type":"polyline",
            "occluded":false,
            "outside":false,
            "z_order":0,
            "rotation":0.0,
            "points":[{_x},{_y - _d/2},{_x},{_y + _d/2}],
            "frame":0,
            "group":0,
            "source":"manual",
            "attributes":[],
            "elements":[],
            "label":"Nanoparticle"
        }}""") 

    shapes = ",\n".join(shapes)

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


# taskCVAT: path to zip file CVAT with labeled particles
def importTaskFromCVAT(taskCVAT):
    with zipfile.ZipFile(taskCVAT, 'r') as tempZipFile:
        annotations = tempZipFile.read('annotations.json')
        annotations = annotations.decode('utf-8')

        manifest = tempZipFile.read('data/manifest.jsonl')
        manifest = manifest.decode('utf-8')
        temp = json.loads(manifest.split('\n')[-1])
        imgFileName = temp['name'] + temp['extension']        
        imageBytes = tempZipFile.read(f'data/{imgFileName}')
    
    particles = []

    annotations = json.loads(annotations)
    for shape in annotations[0]['shapes']:
        points = shape['points']

        coordinates = [[points[0], points[1]], [points[2], points[3]]]

        d = math.dist(coordinates[0], coordinates[1])
        x = (coordinates[0][0] + coordinates[1][0]) / 2
        y = (coordinates[0][1] + coordinates[1][1]) / 2
        
        particles.append([x, y, d])

    return np.array(particles), imgFileName, io.BytesIO(imageBytes)


if __name__ == "__main__": 
    
    from PIL import Image
    taskFile = r"D:\��������\task_nano labeling_backup_2025_09_11_10_57_41.zip"

    temp = importTaskFromCVAT(taskFile)
    print(temp)