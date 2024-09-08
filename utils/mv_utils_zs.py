from torch_scatter import scatter
import torch.nn as nn
import numpy as np
import torch

TRANS = -1.5

# realistic projection parameters
params = {'maxpoolz': 1, 'maxpoolxy': 7, 'maxpoolpadz': 0, 'maxpoolpadxy': 2,
          'convz': 1, 'convxy': 3, 'convsigmaxy': 3, 'convsigmaz': 1, 'convpadz': 0, 'convpadxy': 1,
          'imgbias': 0., 'depth_bias': 0.2, 'obj_ratio': 0.8, 'bg_clr': 0.0,
          'resolution': 112, 'depth': 8}


class Grid2Image(nn.Module):
    """A pytorch implementation to turn 3D grid to 2D image. 将3D网格转换为2D图像
       Maxpool: densifying the grid
       Convolution: smoothing via Gaussian
    """

    def __init__(self):
        super().__init__()
        torch.backends.cudnn.benchmark = False

        self.maxpool = nn.MaxPool3d((params['maxpoolz'], params['maxpoolxy'], params['maxpoolxy']),
                                    stride=1, padding=(params['maxpoolpadz'], params['maxpoolpadxy'],
                                                       params['maxpoolpadxy']))

        self.conv = torch.nn.Conv3d(1, 1, kernel_size=(params['convz'], params['convxy'], params['convxy']),
                                    stride=1, padding=(params['convpadz'], params['convpadxy'], params['convpadxy']),
                                    bias=True)
        kn3d = get3DGaussianKernel(params['convxy'], params['convz'], sigma=params['convsigmaxy'],
                                   zsigma=params['convsigmaz'])
        self.conv.weight.data = torch.Tensor(kn3d).repeat(1, 1, 1, 1, 1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.maxpool(x.unsqueeze(1))
        x = self.conv(x)
        img = torch.max(x, dim=2)[0]
        img = img / torch.max(torch.max(img, dim=-1)[0], dim=-1)[0][:, :, None, None]
        img = 1 - img
        img = img.repeat(1, 3, 1, 1)
        return img


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat


##  将点云转化为网格
def points2grid(points, resolution=params['resolution'], depth=params['depth']):
    """Quantize each point cloud to a 3D grid.
    Args:
        points (torch.tensor): of size [B, _, 3]
    Returns:
        grid (torch.tensor): of size [B * self.num_views, depth, resolution, resolution]
    """

    batch, pnum, _ = points.shape

    # 中心化和标准化
    """
        （1）先求每个批次点云数据的最大值和最小值
        （2）然后通过最大值最小值求出中心
        （3）计算每个批次中点云数据在每个维度上的范围，取最大范围。
        （4）将点云数据中心化并标准化。通过减去中心点并除以范围，将点云数据映射到 [-1, 1] 的范围内。
    """
    pmax, pmin = points.max(dim=1)[0], points.min(dim=1)[0]
    pcent = (pmax + pmin) / 2
    pcent = pcent[:, None, :]
    prange = (pmax - pmin).max(dim=-1)[0][:, None, None]
    points = (points - pcent) / prange * 2.
    ##params['obj_ratio']相当于是论文中的 s
    points[:, :, :2] = points[:, :, :2] * params['obj_ratio']  ##对点云数据的前两个维度（x和y坐标）进行缩放，通过乘以一个比例因子 obj_ratio。

    # 深度偏差
    """"
        (1)计算点云数据在 x,y,z 轴上的量化坐标。首先将 x,y,z 轴坐标映射到 [0, 1] 的范围，然后乘以 resolution 得到在网格中的坐标。
        (2)resolution 代表了网格的空间分辨率
    """
    depth_bias = params['depth_bias']
    _x = (points[:, :, 0] + 1) / 2 * resolution
    _y = (points[:, :, 1] + 1) / 2 * resolution
    _z = ((points[:, :, 2] + 1) / 2 + depth_bias) / (1 + depth_bias) * (depth - 2)

    # 量化坐标
    _x.ceil_()  # _x 和 _y 的量化，使用了 ceil() 函数，即向上取整，将浮点数坐标映射到最接近的整数格子中
    _y.ceil_()
    z_int = _z.ceil()  # _z，在量化时使用了 _z.ceil()，这是因为深度（z 轴坐标）的量化可能需要更精细的控制。在这里，作者选择使用向上取整的方式，确保 _z 被映射到一个整数深度格子。

    # 作用是确保点云在被映射到离散网格时不会超出合理的范围，避免超出网格的边界
    _x = torch.clip(_x, 1, resolution - 2)
    _y = torch.clip(_y, 1, resolution - 2)
    _z = torch.clip(_z, 1, depth - 2)

    coordinates = z_int * resolution * resolution + _y * resolution + _x

    # 创建初始网格
    grid = torch.ones([batch, depth, resolution, resolution], device=points.device).view(batch, -1) * params['bg_clr']

    # 将点云信息分散到网格中
    grid = scatter(_z, coordinates.long(), dim=1, out=grid, reduce="max")

    # 重新排列和调整网格形状
    grid = grid.reshape((batch, depth, resolution, resolution)).permute((0, 1, 3, 2))

    return grid


class Realistic_Projection:
    """For creating images from PC based on the view information.
    """

    def __init__(self):
        """
        大小为5张 ###############################################################
        """
        # 大小为5张
        # # 定义视图信息，包括旋转角度和平移值
        # _views = np.asarray([
        #     [[1 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],   # [[旋转角度_x, 旋转角度_z, 旋转角度y], [平移_x, 平移_z, 平移_y]]
        #     [[7 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     ])
        # # 添加一些偏差以显示更多表面
        # # adding some bias to the view angle to reveal more surface
        # # _views_bias的作用是添加一些偏差（或扰动）到原始视图角度和平移值，以显示点云数据的更多表面细节。这种偏差有助于在生成图像时提供更多的视角，从而使得点云的表面细节更加丰富和清晰。
        #
        # _views_bias = np.asarray([
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],         # [旋转角度偏差_x, 旋转角度偏差_z, 旋转角度偏差_y], [平移偏差_x, 平移偏差_z, 平移偏差_y]
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     ])

        # """
        # 大小为10张 ###############################################################
        # """
        # # 大小为10张
        # # 定义视图信息，包括旋转角度和平移值
        # _views = np.asarray([
        #     [[1 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],   # [[旋转角度_x, 旋转角度_z, 旋转角度y], [平移_x, 平移_z, 平移_y]]
        #     [[3 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[5 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[7 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[2 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     ])
        # # 添加一些偏差以显示更多表面
        # # adding some bias to the view angle to reveal more surface
        # # _views_bias的作用是添加一些偏差（或扰动）到原始视图角度和平移值，以显示点云数据的更多表面细节。这种偏差有助于在生成图像时提供更多的视角，从而使得点云的表面细节更加丰富和清晰。
        #
        # _views_bias = np.asarray([
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],         # [旋转角度偏差_x, 旋转角度偏差_z, 旋转角度偏差_y], [平移偏差_x, 平移偏差_z, 平移偏差_y]
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     ])

        """
        大小为15张 ###############################################################
        """
        # 大小为15张
        # 定义视图信息，包括旋转角度和平移值
        _views = np.asarray([
            [[1 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],  # [[旋转角度_x, 旋转角度_z, 旋转角度y], [平移_x, 平移_z, 平移_y]]
            [[2 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[3 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[4 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[5 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[6 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[7 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[8 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[9 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[10 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[11 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[12 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, -np.pi / 2, 0], [-0.5, -0.5, TRANS]],
            [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        ])
        # 添加一些偏差以显示更多表面
        # adding some bias to the view angle to reveal more surface
        # _views_bias的作用是添加一些偏差（或扰动）到原始视图角度和平移值，以显示点云数据的更多表面细节。这种偏差有助于在生成图像时提供更多的视角，从而使得点云的表面细节更加丰富和清晰。

        _views_bias = np.asarray([
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],  # [旋转角度偏差_x, 旋转角度偏差_z, 旋转角度偏差_y], [平移偏差_x, 平移偏差_z, 平移偏差_y]
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        ])

        """
        大小为20张 ###############################################################
        """
        # 大小为20张
        # 定义视图信息，包括旋转角度和平移值
        # _views = np.asarray([
        #     [[1 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],  # [[旋转角度_x, 旋转角度_z, 旋转角度y], [平移_x, 平移_z, 平移_y]]
        #     [[2 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[3 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[4 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[5 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[6 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[7 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[8 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[9 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[10 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[11 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[12 * np.pi / 6, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[-np.pi / 4, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[-np.pi / 2, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[np.pi / 4, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[np.pi / 2, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[0, -np.pi / 4, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, np.pi / 4, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        # ])
        # # 添加一些偏差以显示更多表面
        # # adding some bias to the view angle to reveal more surface
        # # _views_bias的作用是添加一些偏差（或扰动）到原始视图角度和平移值，以显示点云数据的更多表面细节。这种偏差有助于在生成图像时提供更多的视角，从而使得点云的表面细节更加丰富和清晰。
        #
        # _views_bias = np.asarray([
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],  # [旋转角度偏差_x, 旋转角度偏差_z, 旋转角度偏差_y], [平移偏差_x, 平移偏差_z, 平移偏差_y]
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        # ])

        """
        大小为25张 ###############################################################
        """
        # 大小为25张
        # 定义视图信息，包括旋转角度和平移值
        # _views = np.asarray([
        #     [[1 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],  # [[旋转角度_x, 旋转角度_z, 旋转角度y], [平移_x, 平移_z, 平移_y]]
        #     [[2 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[3 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[4 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[5 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[6 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[7 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[8 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[9 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[10 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[11 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[12 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[13 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[14 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[15 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[16 * np.pi / 8, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #
        #     [[-np.pi / 4, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[-np.pi / 2, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[np.pi / 4, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[np.pi / 2, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #
        #     [[0, -np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #
        #     [[0, -np.pi / 4, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, np.pi / 4, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        #
        # ])
        # # 添加一些偏差以显示更多表面
        # # adding some bias to the view angle to reveal more surface
        # # _views_bias的作用是添加一些偏差（或扰动）到原始视图角度和平移值，以显示点云数据的更多表面细节。这种偏差有助于在生成图像时提供更多的视角，从而使得点云的表面细节更加丰富和清晰。
        #
        # _views_bias = np.asarray([
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],  # [旋转角度偏差_x, 旋转角度偏差_z, 旋转角度偏差_y], [平移偏差_x, 平移偏差_z, 平移偏差_y]
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        # ])

        """
        大小为30张 ###############################################################
        """
        # 大小为30张
        # 定义视图信息，包括旋转角度和平移值
        # _views = np.asarray([
        #     [[1 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],  # [[旋转角度_x, 旋转角度_z, 旋转角度y], [平移_x, 平移_z, 平移_y]]
        #     [[2 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[3 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[4 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[5 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[6 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[7 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[8 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[9 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[10 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[11 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[12 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[13 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[14 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[15 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[16 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[17 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[18 * np.pi / 9, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #
        #     [[-np.pi / 6, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[-2 * np.pi / 6, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[-3 * np.pi / 6, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[np.pi / 6, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[2 * np.pi / 6, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[3 * np.pi / 6, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #
        #     [[0, -np.pi / 6, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, -2 * np.pi / 6, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, -3 * np.pi / 6, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, np.pi / 6, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, 2 * np.pi / 6, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, 3 * np.pi / 6, np.pi / 2], [-0.5, -0.5, TRANS]],
        # ])
        # # 添加一些偏差以显示更多表面
        # # adding some bias to the view angle to reveal more surface
        # # _views_bias的作用是添加一些偏差（或扰动）到原始视图角度和平移值，以显示点云数据的更多表面细节。这种偏差有助于在生成图像时提供更多的视角，从而使得点云的表面细节更加丰富和清晰。
        #
        # _views_bias = np.asarray([
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],  # [旋转角度偏差_x, 旋转角度偏差_z, 旋转角度偏差_y], [平移偏差_x, 平移偏差_z, 平移偏差_y]
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        # ])

        """
        大小为40张 ###############################################################
        """
        # 大小为40张
        # 定义视图信息，包括旋转角度和平移值
        # _views = np.asarray([
        #     [[1 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],  # [[旋转角度_x, 旋转角度_z, 旋转角度y], [平移_x, 平移_z, 平移_y]]
        #     [[2 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[3 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[4 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[5 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[6 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[7 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[8 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[9 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[10 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[11 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[12 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[13 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[14 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[15 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[16 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[17 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],  # [[旋转角度_x, 旋转角度_z, 旋转角度y], [平移_x, 平移_z, 平移_y]]
        #     [[18 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[19 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[20 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[21 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[22 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[23 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[24 * np.pi / 12, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        #
        #     [[-np.pi / 8, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[-2 * np.pi / 8, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[-3 * np.pi / 8, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[-4 * np.pi / 8, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[np.pi / 8, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[2 * np.pi / 8, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[3 * np.pi / 8, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #     [[4 * np.pi / 8, np.pi / 2, 0], [-0.5, -0.5, TRANS]],
        #
        #     [[0, -np.pi / 8, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, -2 * np.pi / 8, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, -3 * np.pi / 8, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, -4 * np.pi / 8, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, np.pi / 8, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, 2 * np.pi / 8, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, 3 * np.pi / 8, np.pi / 2], [-0.5, -0.5, TRANS]],
        #     [[0, 4 * np.pi / 8, np.pi / 2], [-0.5, -0.5, TRANS]],
        #
        # ])
        # # 添加一些偏差以显示更多表面
        # # adding some bias to the view angle to reveal more surface
        # # _views_bias的作用是添加一些偏差（或扰动）到原始视图角度和平移值，以显示点云数据的更多表面细节。这种偏差有助于在生成图像时提供更多的视角，从而使得点云的表面细节更加丰富和清晰。
        #
        # _views_bias = np.asarray([
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],  # [旋转角度偏差_x, 旋转角度偏差_z, 旋转角度偏差_y], [平移偏差_x, 平移偏差_z, 平移偏差_y]
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],  # [旋转角度偏差_x, 旋转角度偏差_z, 旋转角度偏差_y], [平移偏差_x, 平移偏差_z, 平移偏差_y]
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        #
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        #     [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        # ])

        # 将视图数量设置为 _views 的行数
        self.num_views = _views.shape[0]

        # 创建第一个旋转矩阵，将角度转换为弧度，并转置矩阵
        angle = torch.tensor(_views[:, 0, :]).float().cuda()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        # 创建第二个旋转矩阵，将角度转换为弧度，并转置矩阵
        angle2 = torch.tensor(_views_bias[:, 0, :]).float().cuda()
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)

        # 创建平移向量，并添加一个维度，以便后续计算
        self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = self.translation.unsqueeze(1)

        self.grid2image = Grid2Image().cuda()

    def get_img(self, points):
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))

        # 网格化点云数据
        grid = points2grid(points=_points, resolution=params['resolution'], depth=params['depth']).squeeze()
        img = self.grid2image(grid)  # 初始化模型进行前向传播
        return img

    @staticmethod
    def point_transform(points, rot_mat, rot_mat2, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param rot_mat2: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        points = points - translation
        return points


# 生成一个二维高斯核。高斯核通常用于图像处理和卷积运算，它在中心达到最大值，然后逐渐减小。生成高斯核的目的通常是在图像处理中进行模糊、平滑或卷积等操作。
def get2DGaussianKernel(ksize, sigma=0):
    center = ksize // 2  # 计算高斯核的中心
    xs = (np.arange(ksize, dtype=np.float32) - center)  # 生成横向坐标轴
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))  # 计算一维高斯核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]  # 将一维高斯核转换为二维高斯核
    kernel = torch.from_numpy(kernel)  # 将 NumPy 数组转换为 PyTorch 张量
    kernel = kernel / kernel.sum()  # 归一化高斯核，使其总和为 1
    return kernel


def get3DGaussianKernel(ksize, depth, sigma=2, zsigma=2):
    kernel2d = get2DGaussianKernel(ksize, sigma)
    zs = (np.arange(depth, dtype=np.float32) - depth // 2)
    zkernel = np.exp(-(zs ** 2) / (2 * zsigma ** 2))
    kernel3d = np.repeat(kernel2d[None, :, :], depth, axis=0) * zkernel[:, None, None]
    kernel3d = kernel3d / torch.sum(kernel3d)
    return kernel3d




