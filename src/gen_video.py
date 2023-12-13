import cv2
import os

# 图片序列所在文件夹路径
image_folder = '/home/zmw/Graduation_Project/data/data1'

# 视频输出文件名和参数
video_name = 'video1.mp4'
fps = 20  # 视频帧率

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]  # 获取文件夹中所有图片
images.sort(key=lambda x: int(x.split('.')[0]))
# 获取第一张图片的尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 将图片序列写入视频
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
