from torch.autograd import Function     # 有些方法没有自带求导，这就要自己写


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):     # x[128, 800], alpha=0.0
        ctx.alpha = alpha

        return x.view_as(x)         # [128, 800]

    @staticmethod                           # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
    def backward(ctx, grad_output):         # dloss / doutput就是输入的参数grad_output. 这边是实现论文提出的梯度反转
        output = grad_output.neg() * ctx.alpha      # grad_output.neg()取反

        return output, None


