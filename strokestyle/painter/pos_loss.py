import torch


def get_relative_pos(point_vars):
    loss = []
    for points in point_vars:
        dist_vec = []
        for i in range(len(points) - 1):
            dist_vec.append(points[i + 1] - points[i])

        dist = torch.norm(torch.stack(dist_vec), dim=1)

        angle_cos = []
        for i in range(len(dist_vec) - 1):
            cos_val = torch.dot(dist_vec[i], dist_vec[i + 1]) / (dist[i] * dist[i + 1])
            angle_cos.append(cos_val)

        angle_cos = torch.stack(angle_cos)
        tmp = torch.concatenate([dist, angle_cos])
        loss.append(tmp)
    return torch.stack(loss)


if __name__ == "__main__":
    x = torch.tensor([[[0, 0], [1, 1], [2, 2]], [[0, 1], [1, 2], [2, 0]]], dtype=torch.float32)
    print(get_relative_pos(x))
