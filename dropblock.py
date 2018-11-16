import os
import numpy as np
from scipy.stats import binom
import mxnet as mx
from mxnet import autograd
import mxnet.ndarray as nd
import random


class DropBlock(mx.operator.CustomOp):
    def __init__(self, drop_prob, block_size):
        super(DropBlock, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        #block_temp = nd.ones((128,48,7,7))

        self.block_mask = nd.ones((256, 48, 7, 7), ctx=mx.gpu(2))

    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0].asnumpy()
        # get gamma value
        gamma = self._compute_gamma(data.shape[1])
        #print gamma
        # sample from mask
        mask_reduction = self.block_size // 2
        mask_height = data.shape[2] - mask_reduction * 2
        mask_width = data.shape[3] - mask_reduction * 2
        mask_area = mask_height * mask_width
        n = int(mask_area * gamma)
        a = np.arange(0, mask_area)
        #b = random.sample(a,n)
        #print in_data[0].shape
        mask = nd.zeros((data.shape[0], 1, mask_area))
        #print mask.shape

        for i in range(0, data.shape[0]):
            b = random.sample(a, n)
            for j in b:
                mask[i, 0, j] = 1

        mask = mask.reshape([data.shape[0], 1, mask_height, mask_width])
        #print mask.shape
        self.block_mask = self._compute_block_mask(mask)
        out_map = in_data[0] * self.block_mask
        # return out_map
        #print req[0]
        self.assign(out_data[0], req[0], nd.array(out_map))

    def _compute_block_mask(self, mask):
        weight_mat = nd.ones(48*self.block_size*self.block_size)
        weight_mat = weight_mat.reshape(
            [48, 1, self.block_size, self.block_size])
        block_mask = nd.Convolution(data=mask, no_bias=True, weight=weight_mat, num_filter=48, kernel=(
            3, 3), pad=(int(np.ceil(self.block_size)/2+1), int(np.ceil(self.block_size)/2+1)))
        # compute mask area
        delta = self.block_size // 2
        input_height = mask.shape[2] + delta*2
        input_width = mask.shape[3] + delta*2
        height_to_crop = block_mask.shape[2] - input_height
        width_to_crop = block_mask.shape[3] - input_width
        #print height_to_crop
        #print width_to_crop
        if height_to_crop != 0:
            block_mask = block_mask[:, :, :-height_to_crop, :]
        if width_to_crop != 0:
            block_mask = block_mask[:, :, :, :-width_to_crop]
        block_mask = 1 - block_mask
        return block_mask

    def _compute_gamma(self, feat_size):
        if feat_size < self.block_size:
            raise ValueError('input shape con not be smaller than block_size')
        return (self.drop_prob/(self.block_size**2))*((feat_size**2)/((feat_size-self.block_size+1)**2))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        dy = out_grad[0] * self.block_mask

        self.assign(in_grad[0], req[0], dy)


@mx.operator.register("DropBlock")
class DropBlockProp(mx.operator.CustomOpProp):
    def __init__(self, drop_prob, block_size):
        super(DropBlockProp, self).__init__(need_top_grad=True)
        self._drop_prob = float(drop_prob)
        self._block_size = int(block_size)

    def list_arguments(self):

        return ['data']  # ,'label']

    def list_outputs(self):
        return ['dropblock_fea']  # ,'label']

    def create_operator(self, ctx, shapes, dtypes):
        return DropBlock(self._drop_prob, self._block_size)
    '''
    def infer_shape(self, in_shape):

	data_shape = in_shape[0]
    	#label_shape = (in_shape[0][0],)
    	output_shape = in_shape[0]
    	#return [data_shape, label_shape], [output_shape], []
        return in_shape,[output_shape],[]

    '''
