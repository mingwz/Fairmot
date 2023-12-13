import os
import os.path as osp

image_flder = "../dataset/mark/images/train/mark_1/img1"  # 数据集所在位置
imgs = os.listdir(image_flder)
# print(imgs)
train_f = open("./data/mark.train", "w")  # 生成.train文件位置
imgs.sort(key=lambda x: int(x.split('.')[0]))

for img_name in imgs:
    image_path = osp.join(image_flder, img_name)
    save_str = image_path[11:] + "\n"  # 修改imgae_path来控制显示的路径
    print(save_str)
    train_f.write(save_str)

train_f.close()
