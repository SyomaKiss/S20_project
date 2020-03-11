from utils.losses import *
import numpy as np


def test_MAE_bbox_weighted():
    cri = MAE_bbox_weighted(100)

    a = torch.ones(16, 3, 512, 512)
    b = torch.zeros(16, 3, 512, 512)
    bbox = np.array([[10, 10], [10, 20], [20, 10], [20, 20]])
    bbox = torch.tensor(np.repeat(bbox[None, ...], 16, axis=0))
    x = 512 * 512 * 3 * 1 - 300 * 1
    x += 300 * 100
    x /= 512 * 512 * 3
    assert cri(a, b, bbox=bbox).mean() == x

    cri = MAE_bbox_weighted(100, border_weighting='linear', margin=5)
    a = torch.ones(16, 3, 512, 512)
    b = torch.zeros(16, 3, 512, 512)
    bbox = np.array([[10, 10], [10, 20], [20, 10], [20, 20]])
    bbox = torch.tensor(np.repeat(bbox[None, ...], 16, axis=0))

    x = 512 * 512 * 3 * 1 - 300 * 1 - 900 * 1
    x += 300 * 100
    x += 600 * 3 * 20
    x += 4 * 55 * 3 * 20
    x /= 512 * 512 * 3

    assert cri(a, b, bbox=bbox).mean() == x


if __name__ == '__main__':
    test_MAE_bbox_weighted()
    print('all tests are passed')
