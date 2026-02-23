from ..models.warp_network import WarpNetwork


class WarpF3D:
    def __init__(
        self,
        warp_network_cfg,
    ):
        self.warp_net = WarpNetwork(**warp_network_cfg)

    def __call__(self, f_s, x_s, x_d, keep_on_gpu=False):
        out = self.warp_net(f_s, x_s, x_d, keep_on_gpu=keep_on_gpu)
        return out
