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


if __name__ == "__main__":
    x = torch.tensor([[[0, 0], [1, 1], [2, 2]], [[0, 1], [1, 2], [2, 0]]], dtype=torch.float32)
    print(get_relative_pos(x))
