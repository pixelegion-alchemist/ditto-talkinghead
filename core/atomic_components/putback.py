import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask


class PutBackNumpy:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            self.mask_ori_float = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            self.mask_ori_float = mask.astype(np.float32) / 255.0

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        result = mask_warped * frame_warped + (1 - mask_warped) * frame_rgb
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return result


class PutBack:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        self.mask_ori_float = np.ascontiguousarray(mask)[:,:,0]
        self.result_buffer = None

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # Use Cython implementation for blending
        blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)

        return self.result_buffer


class PutBackGPU:
    """GPU-resident putback: all static data pre-cached, only one CPU download at the end.

    Pre-computes per-source-frame:
      - warped mask (affine grid for mask is static)
      - frame_rgb as GPU tensor
      - affine grid for render_image warp

    Per frame, only render_image warp + blend happens on GPU.
    Single .cpu().numpy() at output for Pipecat's OutputImageRawFrame.
    """

    def __init__(self, device="cuda", mask_template_path=None):
        self.device = device

        # Build mask tensor [1, 1, 512, 512]
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)  # [512, 512, 1]
            mask_np = mask[:, :, 0]  # [512, 512]
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            mask_np = mask[:, :, 0]

        # Keep mask on GPU as [1, 1, H, W] for grid_sample
        self.mask_gpu = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)

        # Per-source-frame caches (populated by setup())
        self._warped_masks = []    # pre-warped masks [1, 1, oh, ow] on GPU
        self._frame_rgb_gpu = []   # source frames [1, 3, oh, ow] on GPU
        self._grids = []           # affine grids [1, oh, ow, 2] on GPU
        self._ready = False

    def _make_affine_grid(self, M_c2o_2x3, src_h, src_w, dst_h, dst_w):
        """Build a sampling grid that maps from dst (original frame) coords
        back to src (512x512 crop) coords, matching cv2.warpAffine behavior.

        cv2.warpAffine(src, M, dsize=(dst_w, dst_h)) maps:
            dst_pixel = M @ src_pixel  (forward: crop→original)

        grid_sample needs the inverse: for each dst pixel, where to sample in src.
        So we invert M and normalize to [-1,1] range for grid_sample.
        """
        # M_c2o is 2x3: [R|t] mapping crop→original
        # We need the inverse: original→crop
        M = np.eye(3, dtype=np.float64)
        M[:2, :] = M_c2o_2x3.astype(np.float64)
        M_inv = np.linalg.inv(M)[:2, :]  # 2x3 inverse

        # Build theta for grid_sample: normalize so that grid coords map [-1,1] → pixel coords
        # grid_sample expects theta that maps from [-1,1] output to [-1,1] input
        # For pixel coords: x_src = M_inv @ [x_dst, y_dst, 1]
        # Normalize: x_norm_src = 2*x_src/(src_w-1) - 1, etc.
        # x_norm_dst = 2*x_dst/(dst_w-1) - 1 → x_dst = (x_norm_dst+1)*(dst_w-1)/2

        # Compose: denormalize dst → pixel → M_inv → pixel src → normalize src
        # dst_denorm: [[dst_w-1)/2, 0, (dst_w-1)/2], [0, (dst_h-1)/2, (dst_h-1)/2], [0,0,1]]
        # src_norm: [[2/(src_w-1), 0, -1], [0, 2/(src_h-1), -1], [0,0,1]]

        dst_denorm = np.array([
            [(dst_w - 1) / 2.0, 0, (dst_w - 1) / 2.0],
            [0, (dst_h - 1) / 2.0, (dst_h - 1) / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)

        src_norm = np.array([
            [2.0 / (src_w - 1), 0, -1],
            [0, 2.0 / (src_h - 1), -1],
            [0, 0, 1]
        ], dtype=np.float64)

        M_inv_3x3 = np.eye(3, dtype=np.float64)
        M_inv_3x3[:2, :] = M_inv

        theta_3x3 = src_norm @ M_inv_3x3 @ dst_denorm
        theta = torch.from_numpy(theta_3x3[:2, :]).float().unsqueeze(0).to(self.device)  # [1, 2, 3]

        grid = F.affine_grid(theta, [1, 1, dst_h, dst_w], align_corners=True)  # [1, dst_h, dst_w, 2]
        return grid

    def setup(self, source_info):
        """Pre-cache all static data on GPU. Call once after avatar registration."""
        self._warped_masks = []
        self._frame_rgb_gpu = []
        self._grids = []

        img_rgb_lst = source_info["img_rgb_lst"]
        M_c2o_lst = source_info["M_c2o_lst"]
        mask_h, mask_w = 512, 512  # crop space dimensions

        for i in range(len(img_rgb_lst)):
            frame_rgb = img_rgb_lst[i]  # [oh, ow, 3] uint8
            M_c2o = M_c2o_lst[i]       # [3, 3] or [2, 3]
            oh, ow = frame_rgb.shape[:2]

            M_2x3 = M_c2o[:2, :] if M_c2o.shape[0] >= 2 else M_c2o

            # Pre-compute affine grid (maps dst pixels → src crop pixels)
            grid = self._make_affine_grid(M_2x3, mask_h, mask_w, oh, ow)
            self._grids.append(grid)

            # Pre-warp mask using the grid (static — same every frame)
            warped_mask = F.grid_sample(
                self.mask_gpu, grid, mode='bilinear', padding_mode='zeros', align_corners=True
            ).clamp(0, 1)  # [1, 1, oh, ow]
            self._warped_masks.append(warped_mask)

            # Source frame as GPU tensor [1, 3, oh, ow] float32 [0, 255]
            frame_gpu = torch.from_numpy(frame_rgb.copy()).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            self._frame_rgb_gpu.append(frame_gpu)

        self._ready = True

    def __call__(self, frame_idx, render_gpu, M_c2o=None):
        """GPU putback. Only render_gpu changes per frame.

        Args:
            frame_idx: source frame index for pre-cached static data
            render_gpu: decoder output, GPU tensor [1, C, H, W] float32 in [0,1]
            M_c2o: ignored (pre-cached), kept for API compat

        Returns:
            numpy uint8 [oh, ow, 3] — the only CPU transfer
        """
        grid = self._grids[frame_idx]
        warped_mask = self._warped_masks[frame_idx]
        frame_rgb_gpu = self._frame_rgb_gpu[frame_idx]

        # render_gpu is [1, C, H, W] in [0, 1] from decoder — scale to [0, 255]
        render_255 = render_gpu * 255.0

        # Warp rendered face from crop space to original frame space
        render_warped = F.grid_sample(
            render_255, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [1, 3, oh, ow]

        # Alpha blend: warped_mask * render_warped + (1 - warped_mask) * frame_rgb
        result = warped_mask * render_warped + (1.0 - warped_mask) * frame_rgb_gpu

        # Single GPU→CPU transfer: the final composited frame
        result_np = result[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        return result_np
