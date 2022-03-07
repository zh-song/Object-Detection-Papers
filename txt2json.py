import cv2
import sys,os,tqdm,json


def txt2json(root_path,category_dict, json_path):

    assert os.path.exists(root_path)
    originLabelsDir = os.path.join(root_path, 'labels')                                        
    originImagesDir = os.path.join(root_path, 'images')

    indexes = os.listdir(originImagesDir)

    images, annotations, categories = [],[],[]
    # 用于保存所有数据的图片信息和标注信息
    json_dict = {'images': [], 'annotations': [],'categories': []}
    # 建立类别标签和数字id的对应关系, 类别id从1开始。
    for cate, cid in category_dict.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    for k, index in enumerate(indexes):
        # 支持 png jpg 格式的图片。
        txtFile = index.replace('images','txt').replace('.jpg','.txt').replace('.png','.txt')
        # 读取图像的宽和高
        im = cv2.imread(os.path.join(root_path, 'images/') + index)
        height, width, _ = im.shape

        json_dict['images'].append({'file_name': index,
                                    'id': k,
                                    'width': width,
                                    'height': height})

        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息。
            continue
            # 标注的id
        ann_id_cnt = 1
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从1 start
                cls_id = int(label[0])   
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                json_dict['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    folder = os.path.join(root_path, 'ann_json')
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_name = os.path.join(root_path, 'ann_json/{}'.format(json_path))
    with open(json_name, 'w') as f:
        json.dump(json_dict, f)
        print('Save annotation to {}'.format(json_name))
        

if __name__ == "__main__":

    
    root_path = "/home/zhsong/Code/datasets/VisDrone2019-DET/VisDrone2019-DET-val/"
    json_path = 'val.json'
    category_dict = {'pedestrian': 1, 'people': 2,'bicycle': 3, 'car': 4, 'van': 5, 'truck': 6, 'tricycle': 7,'awning-tricycle': 8, 'bus': 9, 'motor': 10}

    txt2json(root_path,category_dict,json_path)
