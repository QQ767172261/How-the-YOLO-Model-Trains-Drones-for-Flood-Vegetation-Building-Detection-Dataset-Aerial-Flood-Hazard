无人机洪水植被建筑物检测数据集-航拍洪水灾害建筑植被检测数据集，2924张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e60e3d986eb74bb89284a30296aadf1b.png)

图像尺寸:640*640
类别数量:2类
训练集图像数量:2048; 验证集图像数量:593； 测试集图像数量:283
类别名称: 每一类图像数 ，每一类标注数
building -建筑物: 2831,32570
vegetation-植被: 2822,15797
image num: 2924

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/65c110d732e747918f40a6ebf5c37771.png)

# 无人机洪水航拍检测数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1a99fe9507514b64abe98bb692ccf782.png)


| 项目 | 详情 |
| --- | --- |
| 数据集名称 | 无人机洪水植被建筑物检测数据集（航拍洪水灾害建筑植被检测） |
| 图像总数 | 2924张 |
| 标注格式 | YOLO、VOC、COCO三种标注 |
| 图像尺寸 | 640×640 |
| 类别数量 | 2类 |
| 训练集 | 2048张 |
| 验证集 | 593张 |
| 测试集 | 283张 |
| 类别标签&数量 | building-建筑物：图像2831张，标注框32570   vegetation-植被：图像2822张，标注框15797 |

## 应用场景

洪水灾害无人机应急航拍、灾后建筑物损毁评估、洪水淹没区地物分类、灾害范围遥感解译、应急救援地物识别、洪涝灾情智能监测。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/816aa352f6f44bb9a47db88e0c6f412d.png)


## 标签

building建筑物、vegetation植被

## YOLOv11简易训练代码

### 1. dataset.yaml

```
path: ./flood_dataset
train: images/train
val: images/val
test: images/test

names:
  0: building
  1: vegetation
```

### 2. train.py

```
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载yolov11n预训练权重
    model = YOLO("yolo11n.pt")
    # 开始训练
    results = model.train(
        data="dataset.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        workers=4
    )
    print("训练完成！")
    # 模型评估
    metrics = model.val()
    print(metrics.box.map)
```

## 推理代码 predict.py

```
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

# 图片推理
res = model("test.jpg", save=True)
for r in res:
    print(r.boxes.cls, r.boxes.conf)
```


