import os
import time
import cv2
import khandy
import numpy as np
import tempfile
import traceback
from insectid import InsectDetector, InsectIdentifier
from flask import Flask, request, jsonify
import logging
import base64
from collections import OrderedDict
from flask_cors import CORS
import sys

app = Flask(__name__)
# 初始化Flask应用
app.logger.addHandler(logging.StreamHandler(sys.stdout))
CORS(app)


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
    :return: 绘制后的图像和识别结果
    """
    image_for_draw = image.copy()
    image_height, image_width = image.shape[:2]
    results = []
    identifyCount = 0
    # 检测昆虫
    boxes, confs, classes = detector.detect(image)
    if boxes is None or confs is None or classes is None:
        return image_for_draw, results

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
        cropped_results = identifier.identify(cropped)
        if not cropped_results:
            continue

        for result in cropped_results:
            # 将 probability 转换为 float 类型
            probability = float(result['probability'])
            ordered_result = OrderedDict([
                ('chinese_name', result['chinese_name']),
                ('latin_name', result['latin_name']),
                ('probability', probability)
            ])
            results.append(ordered_result)

        prob = cropped_results[0]['probability']
        if prob < 0.01:
            text = 'Unknown'
        else:
            text = '{}: {:.3f}'.format(cropped_results[0]['chinese_name'], cropped_results[0]['probability'])

        # 计算文本位置
        position = [box[0] + 2, box[1] - 20]
        position[0] = np.clip(position[0], 0, image_width)
        position[1] = np.clip(position[1], 0, image_height)
        identifyCount = identifyCount + 1
        # 确保图像数组是可写的
        image_for_draw = image_for_draw.copy()

        # 绘制矩形框
        cv2.rectangle(image_for_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # 绘制文本
        image_for_draw = khandy.draw_text(image_for_draw, text, position, font='simsun.ttc', font_size=15)

    return image_for_draw, results, identifyCount


@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "未上传图像文件"}), 400

    image_file = request.files['image']
    try:
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            return jsonify({"error": "不支持的文件格式"}), 400

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image_file.save(tmp)
            tmp_path = tmp.name

        # 初始化检测器和识别器
        detector = InsectDetector()
        identifier = InsectIdentifier()

        # 读取图像
        image = khandy.imread(tmp_path)
        if image is None:
            return jsonify({"error": "无法读取图像"}), 500

        # 调整图像大小
        if max(image.shape[:2]) > 1280:
            image = khandy.resize_image_long(image, 1280)

        # 处理图像
        image_for_draw, results, identifyCount = process_image(image, detector, identifier)

        # 将处理后的图像转换为 Base64 编码
        _, buffer = cv2.imencode('.jpg', image_for_draw)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        os.unlink(tmp_path)

        return jsonify({"image": image_base64, "results": results, "identifyCount": identifyCount})

    except Exception as e:
        app.logger.error(f"请求处理失败:\n{traceback.format_exc()}")
        return jsonify({"error": "服务器错误"}), 500


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    app.logger.info("启动OCR服务...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
