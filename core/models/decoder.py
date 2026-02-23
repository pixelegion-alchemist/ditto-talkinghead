import numpy as np
import torch
from ..utils.load_model import load_model


class Decoder:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "SPADEDecoder",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, feature, keep_on_gpu=False):
        """
        feature: np.ndarray or torch.Tensor (GPU)
        keep_on_gpu: if True, return GPU tensor [1, C, H, W] float32 instead of numpy
        """
        if self.model_type == "onnx":
            if isinstance(feature, torch.Tensor):
                feature = feature.cpu().numpy()
            pred = self.model.run(None, {"feature": feature})[0]
            if keep_on_gpu:
                return torch.from_numpy(pred).to(self.device)
        elif self.model_type == "tensorrt":
            if isinstance(feature, torch.Tensor):
                feature = feature.cpu().numpy()
            self.model.setup({"feature": feature})
            self.model.infer()
            pred = self.model.buffer["output"][0].copy()
            if keep_on_gpu:
                return torch.from_numpy(pred).to(self.device)
        elif self.model_type == "pytorch":
            with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
                if isinstance(feature, torch.Tensor):
                    f_gpu = feature if feature.is_cuda else feature.to(self.device)
                else:
                    f_gpu = torch.from_numpy(feature).to(self.device)

                pred_gpu = self.model(f_gpu).float()

                if keep_on_gpu:
                    return pred_gpu  # [1, C, H, W] float32 on GPU
                pred = pred_gpu.cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        pred = np.transpose(pred[0], [1, 2, 0]).clip(0, 1) * 255    # [h, w, c]
        return pred
