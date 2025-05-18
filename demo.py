import os
import time
import cv2
import khandy
import numpy as np
from insectid import InsectDetector, InsectIdentifier


def get_image_filenames(src_dirs):
    """
    获取指定目录下的所有图像文件，并按修改时间排序
    :param src_dirs: 源目录列表
    :return: 排序后的图像文件列表
    """
    filenames = sum([khandy.list_files_in_dir(src_dir, True) for src_dir in src_dirs], [])
    return sorted(filenames, key=lambda t: os.stat(t).st_mtime, reverse=True)


def process_image(image, detector, identifier):
    """
    处理单张图像，检测昆虫并识别
    :param image: 输入图像
    :param detector: 昆虫检测器
    :param identifier: 昆虫识别器
    :return: 绘制后的图像
    """
    image_for_draw = image.copy()
    image_height, image_width = image.shape[:2]

    # 检测昆虫
    boxes, confs, classes = detector.detect(image)
    if boxes is None or confs is None or classes is None:
        return image_for_draw

    for box, conf, class_ind in zip(boxes, confs, classes):
        box = box.astype(np.int32)
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        # 过滤过小的框
        if box_width < 30 or box_height < 30:
            continue

        # 裁剪图像
        cropped = khandy.crop_image(image, box[0], box[1], box[2], box[3])

        # 识别昆虫
        results = identifier.identify(cropped)
        if not results:
            continue

        print(results[0])
        prob = results[0]['probability']
        if prob < 0.10:
            text = 'Unknown'
        else:
            text = '{}: {:.3f}'.format(results[0]['chinese_name'], results[0]['probability'])

        # 计算文本位置
        position = [box[0] + 2, box[1] - 20]
        position[0] = np.clip(position[0], 0, image_width)
        position[1] = np.clip(position[1], 0, image_height)

        # 确保图像数组是可写的
        image_for_draw = image_for_draw.copy()

        # 绘制矩形框
        cv2.rectangle(image_for_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # 绘制文本
        image_for_draw = khandy.draw_text(image_for_draw, text, position, font='simsun.ttc', font_size=15)

    return image_for_draw


if __name__ == '__main__':
    src_dirs = [r'images', r'N:\ai\quarrying-insect-id\images']
    result_dir = r'N:\ai\quarrying-insect-id\result'

    # 创建结果目录（如果不存在）
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 初始化检测器和识别器
    detector = InsectDetector()
    identifier = InsectIdentifier()

    # 获取图像文件列表
    src_filenames = get_image_filenames(src_dirs)

    for k, filename in enumerate(src_filenames):
        print(f'[{k + 1}/{len(src_filenames)}] {filename}')
        start_time = time.time()

        # 读取图像
        image = khandy.imread(filename)
        if image is None:
            continue

        # 调整图像大小
        if max(image.shape[:2]) > 1280:
            image = khandy.resize_image_long(image, 1280)

        # 处理图像
        image_for_draw = process_image(image, detector, identifier)

        print(f'Elapsed: {time.time() - start_time:.3f}s')

        # 获取文件名（不含路径）
        file_name = os.path.basename(filename)
        # 构建保存路径
        save_path = os.path.join(result_dir, file_name)
        # 保存图像
        cv2.imwrite(save_path, image_for_draw)