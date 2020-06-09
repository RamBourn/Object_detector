#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/4
"""

"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image
import tkinter as tk
from PIL import Image,ImageTk
import tkinter.ttk
import tkinter.filedialog


class YOLO(object):
    def __init__(self):
        self.anchors_path = 'configs/yolo_anchors.txt'  # Anchors
        self.model_path = 'model_data/yolo_weights.h5'  # 模型文件
        self.classes_path = 'configs/coco_classes_ch.txt'  # 类别文件

        # self.model_path = 'model_data/ep074-loss26.535-val_loss27.370.h5'  # 模型文件
        # self.classes_path = 'configs/wider_classes.txt'  # 类别文件

        self.score = 0.60
        self.iou = 0.45
        # self.iou = 0.01
        self.class_names = self._get_class()  # 获取类别
        self.anchors = self._get_anchors()  # 获取anchor
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw

        self.colors = self.__get_colors(self.class_names)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @staticmethod
    def __get_colors(names):
        # 不同的框，不同的颜色
        hsv_tuples = [(float(x) / len(names), 1., 1.)
                      for x in range(len(names))]  # 不同颜色
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        return colors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)  # 转换~
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)  # anchors的数量
        num_classes = len(self.class_names)  # 类别数

        self.yolo_model = yolo_body(Input(shape=(416, 416, 3)), 3, num_classes)
        self.yolo_model.load_weights(model_path)  # 加载模型参数

        #print('{} model, {} anchors, and {} classes loaded.'.format(model_path, num_anchors, num_classes))

        # 根据检测参数，过滤框
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output, self.anchors, len(self.class_names),
            self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        
        start = timer()  # 起始时间

        if self.model_image_size != (None, None):  # 416x416, 416=32*13，必须为32的倍数，最小尺度是除以32
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维

        # 参数盒子、得分、类别；输入图像0~1，4维；原始图像的尺寸
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
        thickness = (image.size[0] + image.size[1]) // 512  # 厚度
        m=0
        for i, c in reversed(list(enumerate(out_classes))):   
            predicted_class = self.class_names[c]  # 类别
            if class_name_list.get()==predicted_class:
                m+=1
                box = out_boxes[i]  # 框
                score = out_scores[i]  # 执行度
                label = '{:.2f}'.format(score)  # 标签
                draw = ImageDraw.Draw(image)  # 画图
                label_size = draw.textsize(label, font)  # 标签文字
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                #print(label, (left, top), (right, bottom))  # 边框
                results.insert(tk.END,'{},({},{}),({},{})\n'.format(label,left,top,right,bottom))
                if top - label_size[1] >= 0:  # 标签文字
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):  # 画框
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(  # 文字背景
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 文案
                del draw

        end = timer()
        results.insert(tk.END,'检测到{}个{}\n'.format(m,class_name_list.get()))  # 检测出的框
        results.insert(tk.END,'用时:{:.2f}s\n'.format(end - start))  # 检测执行时间
        return image

    def detect_objects_of_image(self, img_path):
        image = Image.open(img_path)
        assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维
        results.insert(tk.END,'detector size {}\n'.format(image_data.shape))

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('out_boxes: {}'.format(out_boxes))
        # print('out_scores: {}'.format(out_scores))
        # print('out_classes: {}'.format(out_classes))

        img_size = image.size[0] * image.size[1]
        objects_line = self._filter_boxes(out_boxes, out_scores, out_classes, img_size)
        return objects_line

    def _filter_boxes(self, boxes, scores, classes, img_size):
        res_items = []
        for box, score, clazz in zip(boxes, scores, classes):
            top, left, bottom, right = box
            box_size = (bottom - top) * (right - left)
            rate = float(box_size) / float(img_size)
            clz_name = self.class_names[clazz]
            if rate > 0.05:
                res_items.append('{}-{:0.2f}'.format(clz_name, rate))
        res_line = ','.join(res_items)
        return res_line

    def close_session(self):
        self.sess.close()


def detect_img_for_test():
    yolo = YOLO()
    img_path = './dataset/class1.jpeg'
    image = Image.open(img_path)
    r_image = yolo.detect_image(image)
    yolo.close_session()
    r_image.save('./results/class1.png')


def test_of_detect_objects_of_image():
    yolo = YOLO()
    img_path = './dataset/vDaPl5QHdoqb2wOaVql4FoJWNGglYk.jpg'
    objects_line = yolo.detect_objects_of_image(img_path)
    print(objects_line)


def img_path_find():
    img_path=tk.filedialog.askopenfilename()
    img_path_en.delete(0,tk.END)
    img_path_en.insert(0,img_path)
    img_path=img_path_en.get()
    img_open=Image.open(img_path)
    img=ImageTk.PhotoImage(img_open.resize((700,500)))
    img_lb.config(image=img)
    img_lb.image=img
    results.delete('1.0',tk.END)

def img_save():
    img_name=os.path.basename(img_path_en.get())
    img_open_r=yolo.detect_image(Image.open(img_path_en.get()))
    img_path_r=os.path.join('./results/',img_name)
    img_open_r.save(img_path_r)

def img_back():
    img_open=Image.open(img_path_en.get())
    img=ImageTk.PhotoImage(img_open.resize((700,500)))
    img_lb.config(image=img)
    img_lb.image=img
    results.delete('1.0',tk.END)

def img_detect():
    img_open=Image.open(img_path_en.get())
    img_open_r=yolo.detect_image(img_open)
    img=ImageTk.PhotoImage(img_open_r.resize((700,500)))
    img_lb.config(image=img)
    img_lb.image=img

def quit():
    yolo.close_session()
    window.quit()

if __name__ == '__main__':
    yolo=YOLO()
    window=tk.Tk()
    window.title("物体检测系统")
    window.geometry("900x580+500+200")
    img_path='./dataset/main.jpeg'
    img_path_r='./dataset/main.jpeg'
    class_name=tk.StringVar()
    img_open=Image.open(img_path)
    img_open_r=Image.open(img_path_r)
    img=ImageTk.PhotoImage(img_open.resize((700,500)))
    img_lb=tk.Label(window,image=img)
    img_path_lb=tk.Label(window,text='图片路径',font=('黑体',14))
    img_path_en=tk.Entry(window,font=('黑体',14),width=1)
    img_path_bt=tk.Button(window,font=('黑体',14),text='...',command=img_path_find)
    class_name_lb=tk.Label(window,text='检测类别',font=('黑体',14))
    class_name_list=tk.ttk.Combobox(window,font=('黑体',14),width=15,textvariable=class_name)
    class_name_ch=open('./configs/coco_classes_ch.txt').readlines()
    for i in range(len(class_name_ch)):
        class_name_ch[i]=class_name_ch[i].strip('\n')
    class_name_en=open('./configs/coco_classes.txt').readlines()
    for i in range(len(class_name_en)):
        class_name_en[i]=class_name_en[i].strip('\n')
    class_name_list['values']=class_name_ch
    save_bt=tk.Button(window,text='保存',font=('黑体',14),command=img_save)
    back_bt=tk.Button(window,text='还原',font=('黑体',14),command=img_back)
    detect_bt=tk.Button(window,text='开始检测',font=('黑体',14),command=img_detect)
    quit_bt=tk.Button(window,text='退出程序',font=('黑体',14),command=quit)
    results=tk.Text(window,font=('黑体',14),width=1,height=2)
    img_lb.grid(row=0,column=0,rowspan=5,columnspan=6,sticky=tk.E+tk.N)
    img_path_lb.grid(row=0,column=6,rowspan=1,columnspan=2,sticky=tk.S)
    img_path_en.grid(row=1,column=6,rowspan=1,columnspan=2,sticky=tk.W+tk.E)
    img_path_bt.grid(row=1,column=7,rowspan=1,columnspan=1,sticky=tk.E)
    class_name_lb.grid(row=2,column=6,rowspan=1,columnspan=2,sticky=tk.S+tk.W+tk.E)
    class_name_list.grid(row=3,column=6,rowspan=1,columnspan=2,sticky=tk.W)
    save_bt.grid(row=4,column=6,rowspan=1,columnspan=1,sticky=tk.S+tk.W+tk.E)
    back_bt.grid(row=4,column=7,rowspan=1,columnspan=1,sticky=tk.S+tk.W+tk.E)
    detect_bt.grid(row=5,column=6,rowspan=1,columnspan=2,sticky=tk.E+tk.W+tk.N+tk.S)
    quit_bt.grid(row=6,column=6,rowspan=1,columnspan=2,sticky=tk.E+tk.W+tk.N+tk.S)
    results.grid(row=5,column=0,rowspan=2,columnspan=6,sticky=tk.E+tk.W+tk.S+tk.N)
    window.mainloop()