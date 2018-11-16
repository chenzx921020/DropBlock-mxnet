# DropBlock
Implementation of DropBlock: A regularization method for convolutional networks in mxnet.

# Usage
## initail parameters
In your symbol,between convolutional operators, you need to previously compute the feature map size and confirm the mask size, this part will be improved in future commit.
```
self.block_mask = nd.ones((256, 48, 7, 7)) # mask size:(batch_size,channel,mask_size,mask_size)
``` 

## operator implementation
In my experiment, feature map size is 7, so I choose drop probability for 0.5 and block size for 3.
```
drop_layer = mx.sym.Custom(conv5,drop_prob=0.5,block_size=3 ,op_type = 'DropBlock')
```

# TODO
> Scheduled DropBlock

> DropBlock for 3D convolution