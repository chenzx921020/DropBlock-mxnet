# DropBlock
Implementation of DropBlock: A regularization method for convolutional networks in mxnet.

# Usage
## initail parameters
In your symbol,between convolutional operators, you need to previously compute the feature map size and confirm the mask size, this part will be improved in future commit.
```
self.block_mask = nd.ones((256, 48, 7, 7)) # mask size:(batch_size,channel,mask_size,mask_size)
``` 

## operator implementation
In my experiment, feature map size is 7, the schedule for drop block probability has been finished, you can set step and prob range in operator.
```
drop_layer = mx.sym.Custom(conv5,drop_prob=0.0,block_size=3,drop_prob_max=0.3,step=15000,block_factor_prob = 0.04 ,op_type = 'DropBlock')
```
For different task, maybe you need to try different parameters for many times.

# TODO

> DropBlock for 3D convolution