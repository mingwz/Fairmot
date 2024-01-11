import torch
import torch.onnx.utils as onnx
from src.lib.models.yolo import get_pose_net


from collections import OrderedDict

model = get_pose_net(num_layers=0, heads={'hm': 1, 'wh': 4, 'reg': 2, 'id': 64}, head_conv=256)

checkpoint = torch.load("../../weights/mark.pth", map_location="cpu")
checkpoint = checkpoint["state_dict"]
change = OrderedDict()
for key, op in checkpoint.items():
    change[key.replace("module.", "", 1)] = op

model.load_state_dict(change)
model.eval()
model.cuda()

input = torch.zeros((1, 3, 608, 1088)).cuda()
[hm, wh, reg, hm_pool, id_feature] = model(input)
onnx.export(model, (input), "../../weights/mark.onnx", input_names=["image"], output_names=["hm", "wh", "reg", "hm_pool", "id"])
