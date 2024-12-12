import os
import numpy as np
import pickle

# 定义数据集的路径
data_path = 'small_data'

# 初始化字典来存储标注数据
gt = {}

# 遍历目录中的所有文件
for filename in os.listdir(data_path):
    if filename.endswith('.txt'):
        # 获取图片文件名
        image_name = filename.replace('.txt', '.png')

        # 构建完整的图片和标注文件路径
        image_path = os.path.join(data_path, image_name)
        txt_path = os.path.join(data_path, filename)

        # 读取标注文件
        with open(txt_path, 'r') as f:
            # 读取所有行
            lines = f.readlines()

            # 初始化边界框列表
            boxes = []

            # 解析每一行，提取边界框信息
            for line in lines:
                # 去除行首和行尾的空格和换行符
                line = line.strip()
                # 按照空格分隔，获取类别、中心坐标和宽高
                class_id, centerx, centery, w, h = map(float, line.split(' '))

                # 计算边界框坐标
                xmin = centerx - w / 2
                ymin = centery - h / 2
                xmax = centerx + w / 2
                ymax = centery + h / 2

                # 添加边界框到列表
                boxes.append([xmin, ymin, xmax, ymax, 1.0])
            # 当边界框列表为空时，生成一个形状为(0,7)的数组
            if not boxes:
                boxes = np.zeros((0, 7), dtype=float)
            # 将边界框列表转换为numpy数组
            boxes = np.array(boxes)

            # 将边界框数组存储在字典中
            gt[image_name] = boxes

# 保存到pkl文件
with open('my_gt_pascal.pkl', 'wb') as f:
    pickle.dump(gt, f)
