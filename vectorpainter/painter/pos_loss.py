import torch
import torch.nn as nn
import random

import vectorpainter.painter.pytorch_batch_sinkhorn as spc


class SinkhornLoss(nn.Module):

    def __init__(self, epsilon=0.01, niter=5, normalize=False, device=None):
        super(SinkhornLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.normalize = normalize
        self.device = device

    def _mesh_grids(self, batch_size, h, w):

        a = torch.linspace(0.0, h - 1.0, h).to(self.device)
        b = torch.linspace(0.0, w - 1.0, w).to(self.device)
        y_grid = a.view(-1, 1).repeat(batch_size, 1, w) / h
        x_grid = b.view(1, -1).repeat(batch_size, h, 1) / w
        grids = torch.cat([y_grid.view(batch_size, -1, 1),
                          x_grid.view(batch_size, -1, 1)], dim=-1)
        return grids

    def forward(self, canvas, gt):

        batch_size, c, h, w = gt.shape
        if h > 24:
            canvas = nn.functional.interpolate(canvas, [24, 24], mode='area')
            gt = nn.functional.interpolate(gt, [24, 24], mode='area')
            batch_size, c, h, w = gt.shape

        canvas_grids = self._mesh_grids(batch_size, h, w)
        gt_grids = torch.clone(canvas_grids)

        # randomly select a color channel, to speedup and consume memory
        i = random.randint(0, 2)

        img_1 = canvas[:, [i], :, :]
        img_2 = gt[:, [i], :, :]

        mass_x = img_1.reshape(batch_size, -1)
        mass_y = img_2.reshape(batch_size, -1)
        if self.normalize:
            loss = spc.sinkhorn_normalized(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)
        else:
            loss = spc.sinkhorn_loss(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)

        return loss


def get_relative_pos(point_vars):
    loss = []
    for points in point_vars:
        dist_vec = points[1:] - points[:-1]  # 计算相邻点之间的向量差
        dist = torch.norm(dist_vec, dim=1)  # 计算向量差的模长
        cos_val = torch.sum(dist_vec[:-1] * dist_vec[1:], dim=1) / (dist[:-1] * dist[1:])  # 计算相邻向量之间的夹角余弦值
        tmp = torch.cat([dist, cos_val])
        loss.append(tmp)
    return torch.stack(loss)


def bezier_curve_loss(curve1, curve2, mean=True):
    # assert curve1.shape == curve2.shape
    loss = []
    for curve_1, curve_2 in zip(curve1, curve2):
        t = torch.linspace(0, 1, curve_1.shape[0]).unsqueeze(1).to(curve_1.device)
        predicted_curve2 = (1 - t) ** 2 * curve_2[0] + 2 * (1 - t) * t * curve_2[1] + t ** 2 * curve_2[2]
        tmp = torch.nn.functional.mse_loss(curve_1, predicted_curve2)
        loss.append(tmp)
    return torch.stack(loss).mean() if mean else torch.stack(loss)


if __name__ == "__main__":
    # shape: [n_curves, n_points, coords]
    x = torch.tensor([[[0, 0], [1, 1], [2, 2]], [[0, 1], [1, 2], [2, 0]]], dtype=torch.float32)
    y = torch.tensor([[[3, 3], [4, 4], [5, 5]], [[3, 4], [5, 6], [7, 8]]], dtype=torch.float32)
    print(get_relative_pos(x))
    print(bezier_curve_loss(x, y, mean=False))
