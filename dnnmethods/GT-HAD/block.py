import torch
import torch.nn.functional as F
import torch.nn as nn 
import pdb

# adopt a sliding window to partition the HSI into N overlapped HSI blocks (cubes)
class Block_embedding(nn.Module):
    def __init__(self, wsize=15, wstride=5):
        super(Block_embedding, self).__init__()
        self.ksize = wsize
        self.stride = wstride

    def same_padding(self, images, ksizes, strides, rates=(1, 1)):
        assert len(images.size()) == 4
        batch_size, channel, rows, cols = images.size()
        out_rows = (rows + strides[0] - 1) // strides[0]
        out_cols = (cols + strides[1] - 1) // strides[1]
        effective_k_row = (ksizes[0] - 1) * rates[0] + 1
        effective_k_col = (ksizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
        # pad the input
        padding_top = int(padding_rows / 2.)
        padding_left = int(padding_cols / 2.)
        padding_bottom = padding_rows - padding_top
        padding_right = padding_cols - padding_left
        # The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward.
        # For example, to pad only the last dimension of the input tensor, then pad has the form (padding_left,padding_right)
        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        images = F.pad(images, paddings, mode='replicate') # replicate reflect 

        return images, paddings

    def extract_image_blocks(self, images, ksizes, strides):
        assert len(images.size()) == 4
        images, paddings = self.same_padding(images, ksizes, strides)
        unfold = torch.nn.Unfold(kernel_size=ksizes,
                                 padding=0,
                                 stride=strides)
        blocks = unfold(images)

        return blocks, paddings

    def forward(self, x):
        _, band, row, col = x.size()
        
        # blocks oder: from left to right, and from top to down
        blocks, paddings = self.extract_image_blocks(x, ksizes=[self.ksize, self.ksize],
                    strides=[self.stride, self.stride],) # batch=1, c*h*w, num
        blocks = blocks.squeeze(0).permute(1, 0) # num, l=c*h*w
        blocks = blocks.view(blocks.size(0), band, self.ksize, self.ksize)

        return blocks, blocks, paddings

# fold operation: combine all blocks back to the original HSI
class Block_fold(nn.Module):
    def __init__(self, wsize=15, wstride=5):
        super(Block_fold, self).__init__()
        self.ksize = wsize
        self.stride = wstride

    def forward(self, x, paddings, row, col):
        num = x.size(0) # num, c, h, w
        back = x.view(num, -1).permute(1, 0).unsqueeze(0) # batch=1, c*h*w, num

        # judge padding value
        block_size1 = (row + 2 * paddings[2] - (self.ksize - 1) - 1) / self.stride + 1
        block_size2 = (col + 2 * paddings[0] - (self.ksize - 1) - 1) / self.stride + 1
        if block_size1 * block_size2 != num:
            pad = [paddings[3], paddings[1]]
        else:
            pad = [paddings[2], paddings[0]]

        try:
            ori = F.fold(back, (row, col), (self.ksize, self.ksize), padding=pad, stride=self.stride)
        except:
            pdb.set_trace()

        # use the average operation to deal with the overlapped areas among blocks. 
        tmp = torch.ones_like(ori)
        tmp_unfold = F.unfold(tmp, (self.ksize, self.ksize), padding=pad, stride=self.stride)
        fold_mask = F.fold(tmp_unfold, (row, col), (self.ksize, self.ksize), padding=pad, stride=self.stride)
        out = ori / fold_mask

        return out
