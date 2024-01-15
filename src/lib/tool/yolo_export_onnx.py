import os
import torch
import torch.onnx.utils as onnx
from src.lib.models.yolo import get_pose_net
from src.lib.models.net import get__net
import tensorrt as trt
from collections import OrderedDict


trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, '')
model = get_pose_net(num_layers=0, heads={'hm': 1, 'wh': 4, 'reg': 2, 'id': 64}, head_conv=256)

checkpoint = torch.load("../../weights/mark.pth", map_location="cpu")
checkpoint = checkpoint["state_dict"]
change = OrderedDict()
for key, op in checkpoint.items():
    change[key.replace("module.", "", 1)] = op

model.load_state_dict(change)
model.eval()
model.cuda()

buffer = []

input2 = torch.ones((1, 3, 608, 1088), dtype=torch.float32, device="cuda")
input3 = torch.ones((1, 3, 608, 1088), dtype=torch.float32, device="cuda")
out_hm = torch.ones((1, 1, 152, 272), dtype=torch.float32, device="cuda")
out_wh = torch.ones((1, 152, 272, 4), dtype=torch.float32, device="cuda")
out_reg = torch.ones((1, 152, 272, 2), dtype=torch.float32, device="cuda")
out_hm_pool = torch.ones((1, 1, 152, 272), dtype=torch.float32, device="cuda")
out_id_feature = torch.ones((1, 152, 272, 64), dtype=torch.float32, device="cuda")

buffer.append(input3.data_ptr())
buffer.append(out_hm.reshape(-1).data_ptr())
buffer.append(out_hm_pool.reshape(-1).data_ptr())
buffer.append(out_id_feature.reshape(-1).data_ptr())
buffer.append(out_wh.reshape(-1).data_ptr())
buffer.append(out_reg.reshape(-1).data_ptr())


[hm, wh, reg, hm_pool, id_feature] = model(input2)
onnx.export(model, (input2), "../../weights/net.onnx", input_names=["image"], output_names=["hm", "wh", "reg", "hm_pool", "id"])

# os.system("trtexec --onnx=../../weights/net.onnx --saveEngine=net.trt")

with open("../../weights/net.trt", 'rb') as f:
    engine_str = f.read()
engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_str)
context = engine.create_execution_context()

context.execute_v2(buffer)

print(torch.allclose(hm, out_hm, rtol=1e-3, atol=1e-2))
print(torch.allclose(wh, out_wh, rtol=1e-3, atol=1e-2))
print(torch.allclose(reg, out_reg, rtol=1e-3, atol=1e-2))
print(torch.allclose(hm_pool, out_hm_pool, rtol=1e-3, atol=1e-2))
print(torch.allclose(id_feature, out_id_feature, rtol=1e-3, atol=1e-2))
