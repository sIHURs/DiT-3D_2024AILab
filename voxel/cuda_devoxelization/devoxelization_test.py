from torch.autograd import Function
import torch

import project_devoxelization

__all__ = ['trilinear_devoxelize']


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, R, R, R]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()
        outs, inds, wgts = project_devoxelization.trilinear_devoxelize_forward(resolution, is_training, coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: 
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = project_devoxelization.trilinear_devoxelize_backward(grad_output.contiguous(), inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r), None, None, None


trilinear_devoxelize = TrilinearDevoxelization.apply


if __name__ == '__main__':
    '''
    N = 65536; F = 256
    rand = torch.rand(N, 8, F, device='cuda')
    feats = rand.clone().requires_grad_()
    feats2 = rand.clone().requires_grad_()
    points = torch.rand(N, 3, device='cuda')*2-1

    t = time.time()
    out_cuda = Trilinear_interpolation_cuda.apply(feats2, points)
    torch.cuda.synchronize()
    print('   cuda fw time', time.time()-t, 's')


    out_cuda = Trilinear_interpolation_cuda.apply(feats2, points)
    '''

    x = torch.randn(1, 3, 32, 32, 32).to('cuda')
    coords = torch.randn(1, 3, 2000).to('cuda')
    output = TrilinearDevoxelization.apply(x, coords, 32)
    print(output.shape)