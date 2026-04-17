from utils import *
import torch
import onnx
from onnx import shape_inference
import onnxruntime as ort
import numpy as np

class LandmarkInferenceModel(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.backbone = trained_model.backbone
        self.pooling = trained_model.pooling
        self.neck = trained_model.neck

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pooling(features)
        embeddings = self.neck(pooled)
        return torch.nn.functional.normalize(embeddings, dim=1)


model = LandmarkModel(model_name='tf_efficientnetv2_s', num_classes=12015, emb_dim=512)
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu"))["model"])
model.eval()


inference_model = LandmarkInferenceModel(model)
inference_model.eval()

dummy_input = torch.randn(1, 3, 384, 384)
torch.onnx.export(
    inference_model, 
    dummy_input, 
    "inference_model.onnx",
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['embeddings'],
    dynamic_axes={'input': {0: 'batch_size'}, 'embeddings': {0: 'batch_size'}},
    opset_version=14
)

model_path = "inference_model.onnx"
onnx_model = onnx.load(model_path)
onnx_model = shape_inference.infer_shapes(onnx_model)
onnx.save(onnx_model, model_path)

inference_model.eval()
with torch.no_grad():
    torch_out = inference_model(dummy_input).numpy()

session = ort.InferenceSession("inference_model.onnx")
onnx_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
onnx_out = session.run(None, onnx_inputs)[0]

np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05)
print("Export verified! Numerical parity achieved.")