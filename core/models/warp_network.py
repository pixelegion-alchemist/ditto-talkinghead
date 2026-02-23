import numpy as np
import torch
from ..utils.load_model import load_model


class WarpNetwork:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "WarpingNetwork",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, feature_3d, kp_source, kp_driving, keep_on_gpu=False):
        """
        feature_3d: np.ndarray or torch.Tensor, shape (1, 32, 16, 64, 64)
        kp_source | kp_driving: np.ndarray, shape (1, 21, 3)
        keep_on_gpu: if True, return GPU tensor instead of numpy (avoids CUDA sync)
        """
        if self.model_type == "onnx":
            if isinstance(feature_3d, torch.Tensor):
                feature_3d = feature_3d.cpu().numpy()
            pred = self.model.run(None, {"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})[0]
        elif self.model_type == "tensorrt":
            if isinstance(feature_3d, torch.Tensor):
                feature_3d = feature_3d.cpu().numpy()
            self.model.setup({"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})
            self.model.infer()
            pred = self.model.buffer["out"][0].copy()
        elif self.model_type == "pytorch":
            with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
                # Accept pre-cached GPU tensor for feature_3d
                if isinstance(feature_3d, torch.Tensor):
                    f3d_gpu = feature_3d if feature_3d.is_cuda else feature_3d.to(self.device)
                else:
                    f3d_gpu = torch.from_numpy(feature_3d).to(self.device)

                pred = self.model(
                    f3d_gpu,
                    torch.from_numpy(kp_source).to(self.device),
                    torch.from_numpy(kp_driving).to(self.device)
                ).float()

                if not keep_on_gpu:
                    pred = pred.cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return pred
