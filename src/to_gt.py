import os
import xml.etree.ElementTree as ET
import sys

if __name__ == "__main__":
    xmls_path = "/home/zmw/Graduation_Project/data/label1/"
    target_path = "/home/zmw/Graduation_Project/data/"

    f = open(target_path + "/" + 'gt' + ".txt", 'w')
    i = 0
    path_list = os.listdir(xmls_path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))
    for xmlFilePath in path_list:
        print(os.path.join(xmls_path, xmlFilePath))
        try:
            tree = ET.parse(os.path.join(xmls_path, xmlFilePath))

            # 获得根节点
            root = tree.getroot()
        except Exception as e:  # 捕获除与程序退出sys.exit()相关之外的所有异常
            print("parse test.xml fail!")
            sys.exit()

        i += 1
        for object in root.iter('object'):
            name = object.find('name')
            if name.text == 'mark_1':
                item = 1
            elif name.text == 'mark_2':
                item = 2

            bndbox = object.find('bndbox')
            # for bndbox in root.iter('bndbox'):
            node = []
            for child in bndbox:
                node.append(int(child.text))
            xmin, ymin = node[0], node[1]
            xmax, ymax = node[2], node[3]
            width = xmax - xmin
            height = ymax - ymin
            # print(xmin, ymin, xmax, ymax, width, height)
            # cat = str(1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1)
            string = str(i) + ',' + str(item) + ',' + str(xmin) + ',' + str(ymax) + ',' + str(width) + ',' + str(
                height) + ',' + str(1)
            # print(string)
            f.write(string + '\n')

    f.close()