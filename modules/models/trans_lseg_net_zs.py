import math
import types
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block as TransformerBlock

from .lseg_blocks_zs import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd

import os

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out

class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x

# tanh relu
class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PositionEmbeddingSine(N_steps, normalize=True):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        elif scale is not None and normalize is False:
            raise ValueError("`normalize` should be True if `scale` is passed")
        self.scale = scale
    
    def forward(self, x, mask=False):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(22, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t//2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # __repr_indent = 4
        lines = [head] + [" " + _repr_indent + line for line in body]
        return "\n".join(lines)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.d_model = d_model

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory):
        """
        tgt: image_features
        memory: text_features
        """
        # add self_attn
        tgt = self.self_attn(query=tgt, 
                            key=tgt, 
                            value=tgt)[0]
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        # add cross_attn
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt),
                                   key=self.with_pos_embed(memory),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers).float()
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)
            if self.norm:
                output = self.norm(output)
        
        return output.unsqueeze(0)


class TransLSeg(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(TransLSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clipRN50x4_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
            "clipRN50x16_vitb32_384": [2, 5, 8, 11],
            "clipRN50x4_vitb32_384": [2, 5, 8, 11],
            "clip_resnet101": [0, 1, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            self.use_pretrained,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # self.scratch.output_conv = head

        
        self.auxlayer = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
        # cosine similarity as logits
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

        if backbone in ["clipRN50x16_vitl16_384", "clipRN50x16_vitb32_384"]:
            self.out_c = 768
        elif backbone in ["clipRN50x4_vitl16_384", "clipRN50x4_vitb32_384"]:
            self.out_c = 640
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        ## add transformer
        hidden_dim = self.out_c
        self.num_decoder_layers = 3
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu")
        # decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_norm = None
        self.decoder = TransformerDecoder(decoder_layer, self.num_decoder_layers, norm=decoder_norm)

        self.arch_option = kwargs["arch_option"]

        self.scratch.output_conv = head

        self.texts = []
        # original
        label = ['others', '']
        # label_list= [a,b,c,d,e]
        # i=1, label = [others, a]
        # i=2, label = [others, b]
        for class_i in range(len(self.label_list)):
            label[1] = self.label_list[class_i]
            text = clip.tokenize(label) #[2,512]
            self.texts.append(text)
        
        # Build Positional Encoding for image features
        N_steps = self.out_c // 2
        self.pos = PositionEmbeddingSine(N_steps, normalize=True)

    def forward(self, x, class_info):
        texts = [self.texts[class_i] for class_i in class_info]
        
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        # 1. Get activations of pretrain
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        # 2. Get feature extraction
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 3. Feature Fusion
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # self.logit_scale = self.logit_scale.to(x.device)
        # 4. Get text encodings via clip pretrained
        text_features = [self.clip_pretrained.encode_text(text.to(x.device)) \
            for text in texts] #len(text_features)=bsz=4, text_features[0].shape=(2, 512)

        # 5. Get image features from last layer
        image_features = self.scratch.head1(path_1) #shape=(bsz, self.out_c, height, width) (4,512,240,240)

        imshape = image_features.shape
        # bsz, h, w, _ = imshape

        image_features = [image_features[i].unsqueeze(0).permute(0,2,3,1).reshape(-1, 1, self.out_c) \
            for i in range(len(image_features))] # permute to (bsz, h, w, out_c); reshape to (h*w, bsz, out_c) = (hw, bsz, 512)
        text_features = [text_features[i].unsqueeze(1).float() \
            for i in range(len(text_features))] # permute to (2, bsz, out_c)
        # print("images[0].shape: ", image_features[0].shape, image_features[0].dtype)
        # print("text[0].shape: ", text_features[0].shape, text_features[0].dtype)

        # mini_batch n_class=4
        # image #1: classes = [a,b]
        # image #2: classes = [c,d]
        # image #3: classes = [a,c]
        # batch_size = 3, n_class=3
        # classes = [a,b,c,d] = label_list
        # text_features = list([2,512],[2,512],[2,512],[2,512])

        # normalized features
        # image_features = [image_feature / image_feature.norm(dim=-1, keepdim=True) for image_feature in image_features]
        # text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]

        # 6. Transformer (query=text, k=v=image)
        logits_per_images = [self.decoder(text_feature, image_feature) \
            for image_feature, text_feature in zip (image_features, text_features)]
            # decoder(tgt=text_feature, memory=image_feature)
            # logits_per_images[0].shape = (1, 1, 2, 512)
        # (hw, 512) * (2, 512)
        # print(logits_per_images[0].shape, "\t", logits_per_images[0].dtype) # 1,2,1,512
        # print(image_features[0].shape, "\t", image_features[0].dtype)  # 57600, 1, 512

        # 7. Do matmul between image features and queried text features
        logits_per_images = [torch.matmul(image_feature.squeeze(1), logit.squeeze().t()) \
            for image_feature, logit in zip(image_features, logits_per_images)]
        # print("logits_per_images[0].shape: ", logits_per_images[0].shape, "\t", logits_per_images[0].dtype)
        # exit()

        # Permute to original image feature shape
        outs = [logits_per_image.float().view(1, imshape[2], imshape[3], -1).permute(0,3,1,2) \
            for logits_per_image in logits_per_images] 
            # outs[0].shape = (bsz, 512, 240, 240)
        out = torch.cat([out for out in outs], dim=0)

        out = self.scratch.output_conv(out) # out.shape = (bsz, 512, 480, 480)
            
        return out


class TransLSegNetZS(TransLSeg):
    """Network for semantic segmentation."""
    def __init__(self, label_list, path=None, scale_factor=0.5, aux=False, use_relabeled=False, use_pretrained=True, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.scale_factor = scale_factor
        self.aux = aux
        self.use_relabeled = use_relabeled
        self.label_list = label_list
        self.use_pretrained = use_pretrained

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)


class TransLSegRN(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_resnet101",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(TransLSegRN, self).__init__()

        self.channels_last = channels_last

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            self.use_pretrained,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # self.scratch.output_conv = head

        self.auxlayer = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
        # cosine similarity as logits
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

        if backbone in ["clipRN50x16_vitl16_384", "clipRN50x16_vitb32_384"]:
            self.out_c = 768
        elif backbone in ["clipRN50x4_vitl16_384", "clipRN50x4_vitb32_384"]:
            self.out_c = 640
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]

        self.scratch.output_conv = head

        self.texts = []
        # original
        label = ['others', '']
        for class_i in range(len(self.label_list)):
            label[1] = self.label_list[class_i]
            text = clip.tokenize(label)
            self.texts.append(text)

    def forward(self, x, class_info):
        texts = [self.texts[class_i] for class_i in class_info]
        
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        self.logit_scale = self.logit_scale.to(x.device)
        text_features = [self.clip_pretrained.encode_text(text.to(x.device)) for text in texts]

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = [image_features[i].unsqueeze(0).permute(0,2,3,1).reshape(-1, self.out_c) for i in range(len(image_features))]

        # normalized features
        image_features = [image_feature / image_feature.norm(dim=-1, keepdim=True) for image_feature in image_features]
        text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]
        
        logits_per_images = [self.logit_scale * image_feature.half() @ text_feature.t() for image_feature, text_feature in zip(image_features, text_features)]
        outs = [logits_per_image.float().view(1, imshape[2], imshape[3], -1).permute(0,3,1,2) for logits_per_image in logits_per_images]
        out = torch.cat([out for out in outs], dim=0)

        out = self.scratch.output_conv(out)
            
        return out


class TransLSegRNNetZS(TransLSegRN):
    """Network for semantic segmentation."""
    def __init__(self, label_list, path=None, scale_factor=0.5, aux=False, use_relabeled=False, use_pretrained=True, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.scale_factor = scale_factor
        self.aux = aux
        self.use_relabeled = use_relabeled
        self.label_list = label_list
        self.use_pretrained = use_pretrained

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Activation should be relu/gelu/glu, not '{activation}'.")