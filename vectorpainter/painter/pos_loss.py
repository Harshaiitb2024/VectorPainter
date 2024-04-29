import torch


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
