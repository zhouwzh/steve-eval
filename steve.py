from utils import *

from dvae import dVAE
from transformer import TransformerEncoder, TransformerDecoder


class SlotAttentionVideo(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size,
                 num_predictor_blocks=1,
                 num_predictor_heads=4,
                 dropout=0.1,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))
        self.predictor = TransformerEncoder(num_predictor_blocks, slot_size, num_predictor_heads, dropout)

    def forward(self, inputs):
        B, T, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        # loop over frames
        attns_collect = []
        slots_collect = []
        for t in range(T):
            # corrector iterations
            for i in range(self.num_iterations):
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
                attn_logits = torch.bmm(k[:, t], q.transpose(-1, -2))
                attn_vis = F.softmax(attn_logits, dim=-1)
                # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

                # Weighted mean.
                attn = attn_vis + self.epsilon
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)
                updates = torch.bmm(attn.transpose(-1, -2), v[:, t])
                # `updates` has shape: [batch_size, num_slots, slot_size].

                # Slot update.
                slots = self.gru(updates.view(-1, self.slot_size),
                                 slots_prev.view(-1, self.slot_size))
                slots = slots.view(-1, self.num_slots, self.slot_size)

                # use MLP only when more than one iterations
                if i < self.num_iterations - 1:
                    slots = slots + self.mlp(self.norm_mlp(slots))

            # collect
            attns_collect += [attn_vis]
            slots_collect += [slots]

            # predictor
            slots = self.predictor(slots)

        attns_collect = torch.stack(attns_collect, dim=1)   # B, T, num_inputs, num_slots
        slots_collect = torch.stack(slots_collect, dim=1)   # B, T, num_slots, slot_size

        return slots_collect, attns_collect


class LearnedPositionalEmbedding1D(nn.Module):

    def __init__(self, num_inputs, input_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, num_inputs, input_size), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input, offset=0):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, offset:offset + T])


class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


class STEVEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.cnn = nn.Sequential(
            Conv2dBlock(args.img_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
        )

        self.cnn_backbone = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2dBlock(256, args.d_model, 5, 1, 2),    #nn.Conv2d + relu()
        )
        # self.cnn = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     Conv2dBlock(2048, 1024, 3, 1, 1),  
        #     Conv2dBlock(1024, 512, 3, 1, 1),   
        #     Conv2dBlock(512, args.d_model, 3, 1, 1),  
        # )

        self.pos = CartesianPositionalEmbedding(args.d_model, args.image_size if args.image_size == 64 else args.image_size // 2)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.mlp = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.d_model, args.d_model))

        self.savi = SlotAttentionVideo(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size,
            args.num_predictor_blocks, args.num_predictor_heads, args.predictor_dropout)

        self.slot_proj = linear(args.slot_size, args.d_model, bias=False)


class STEVEDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dict = OneHotDictionary(args.vocab_size, args.d_model)

        self.bos = nn.Parameter(torch.Tensor(1, 1, args.d_model))
        nn.init.xavier_uniform_(self.bos)

        self.pos = LearnedPositionalEmbedding1D(1 + (args.image_size // 4) ** 2, args.d_model)

        self.tf = TransformerDecoder(
            args.num_decoder_blocks, (args.image_size // 4) ** 2, args.d_model, args.num_decoder_heads, args.dropout)

        self.head = linear(args.d_model, args.vocab_size, bias=False)

class ResNextWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dataset == "movie":
            model = torchvision_models.__dict__[args.backbone_arch](num_classes=0)  #resnext50_32x4d
            state_dict = torch.load(args.backbone_checkpoint, map_location="cpu")
            state_dict = state_dict['teacher']
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            from eminorhan import utils as eminorhan_utils
            model = eminorhan_utils.load_model(model_name='dino_s_resnext50')


        # model.maxpool = nn.Identity()  # remove maxpool layer
        # blk3 = model.layer3[0]
        # blk3.conv2.stride = (1, 1)  # change stride to 1 for layer3
        # blk3.downsample[0].stride = (1, 1)  # change stride to 1 for downsample in layer3
        # blk4 = model.layer4[0]
        # blk4.conv2.stride = (1, 1)  # change stride to 1 for layer4
        # blk4.downsample[0].stride = (1, 1)  # change stride to 1 for downsample in layer4

        self.backbone = model
        self._features = {}
        self.backbone.layer4.register_forward_hook(self._hook_fn)
        self.backbone.layer1.register_forward_hook(self._hook_fn1)
        self.backbone.layer2.register_forward_hook(self._hook_fn2)
        self.backbone.layer3.register_forward_hook(self._hook_fn3)
    
    def _hook_fn(self, module, input, output):
        self._features['layer4'] = output
    def _hook_fn1(self, module, input, output):
        self._features['layer1'] = output
    def _hook_fn2(self, module, input, output):
        self._features['layer2'] = output
    def _hook_fn3(self, module, input, output):
        self._features['layer3'] = output
    

    def forward(self, x):
        _ = self.backbone(x)
        # import pdb; pdb.set_trace()
        return self._features['layer1']
        


from torchvision import models as torchvision_models
import os
class STEVE(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.cnn_hidden_size = args.cnn_hidden_size
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.img_channels = args.img_channels
        self.image_size = args.image_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model

        # dvae
        self.dvae = dVAE(args.vocab_size, args.img_channels)

        # encoder networks
        self.steve_encoder = STEVEEncoder(args)

        # decoder networks
        self.steve_decoder = STEVEDecoder(args)

        self.use_dvae = args.use_dvae
        self.use_backbone = args.use_backbone
        if args.use_backbone:
            self.backbone = ResNextWrapper(args)

    def forward(self, video, tau, hard):
        B, T, C, H, W = video.size()

        video_flat = video.flatten(end_dim=1)                               # B * T, C, H, W
        if self.use_backbone:
            backbone_feature = self.backbone(video_flat)   # B*T,2048,4,4
            # 72, 256, 32, 32
            # 72, 512, 16, 16
            # 72, 1024, 8, 8
            # 72, 2048, 4, 4

        # dvae encode
        if self.use_dvae:
            z_logits = F.log_softmax(self.dvae.encoder(video_flat), dim=1)       # B * T, vocab_size, H_enc, W_enc
        else:
            z_logits = F.log_softmax(backbone_feature, dim=1)
        z_soft = gumbel_softmax(z_logits, tau, hard, dim=1)                  # B * T, vocab_size, H_enc, W_enc
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()         # B * T, vocab_size, H_enc, W_enc
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                         # B * T, H_enc * W_enc, vocab_size
        z_emb = self.steve_decoder.dict(z_hard)                                                     # B * T, H_enc * W_enc, d_model
        z_emb = torch.cat([self.steve_decoder.bos.expand(B * T, -1, -1), z_emb], dim=1)             # B * T, 1 + H_enc * W_enc, d_model
        z_emb = self.steve_decoder.pos(z_emb)                                                       # B * T, 1 + H_enc * W_enc, d_model

        # dvae recon
        if self.use_dvae:
            dvae_recon = self.dvae.decoder(z_soft).reshape(B, T, C, H, W)               # B, T, C, H, W
            dvae_mse = ((video - dvae_recon) ** 2).sum() / (B * T)                      # 1
        else:
            dvae_recon = torch.randn(B, T, C, H, W).to(device=video.device)
            dvae_mse = torch.tensor(0.0, device=video.device)

        # savi
        if self.use_backbone:
            emb = self.steve_encoder.cnn_backbone(backbone_feature)      # B * T, cnn_hidden_size, H, W
        else:
            emb = self.steve_encoder.cnn(video_flat)
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)         # slots: B, T, num_slots, slot_size
                                                                # attns: B, T, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)          # B, T, num_slots, 1, H, W
        attns = video.unsqueeze(2) * attns + (1. - attns)                               # B, T, num_slots, C, H, W

        # decode
        slots = self.steve_encoder.slot_proj(slots)                                                         # B, T, num_slots, d_model
        pred = self.steve_decoder.tf(z_emb[:, :-1], slots.flatten(end_dim=1))                               # B * T, H_enc * W_enc, d_model
        pred = self.steve_decoder.head(pred)                                                                # B * T, H_enc * W_enc, vocab_size
        cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / (B * T)                         # 1

        # if self.use_dvae:
        #     return (dvae_recon.clamp(0., 1.),cross_entropy,dvae_mse,attns)
        # else:
        return (dvae_recon,cross_entropy,dvae_mse,attns)

    def encode(self, video):
        B, T, C, H, W = video.size()

        video_flat = video.flatten(end_dim=1)
        if self.use_backbone:
            backbone_feature = self.backbone(video_flat)   # B*T,2048,4,4

        # savi
        # emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
        if self.use_backbone:
            emb = self.steve_encoder.cnn_backbone(backbone_feature)
        else:
            emb = self.steve_encoder.cnn(video_flat)
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)     # slots: B, T, num_slots, slot_size
                                                            # attns: B, T, num_slots, num_inputs

        attns = attns \
            .transpose(-1, -2) \
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc) \
            .repeat_interleave(H // H_enc, dim=-2) \
            .repeat_interleave(W // W_enc, dim=-1)                      # B, T, num_slots, 1, H, W

        attns_vis = video.unsqueeze(2) * attns + (1. - attns)           # B, T, num_slots, C, H, W

        return slots, attns_vis, attns

    def decode(self, slots):
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        slots = self.steve_encoder.slot_proj(slots)

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.steve_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.steve_decoder.tf(
                self.steve_decoder.pos(input),
                slots
            )
            z_next = F.one_hot(self.steve_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.steve_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer.clamp(0., 1.)

    def reconstruct_autoregressive(self, video):
        """
        image: batch_size x img_channels x H x W
        """
        B, T, C, H, W = video.size()
        slots, attns, _ = self.encode(video)
        recon_transformer = self.decode(slots.flatten(end_dim=1))
        recon_transformer = recon_transformer.reshape(B, T, C, H, W)

        return recon_transformer

    def get_slot(self,image): # input B,C,H,W
        image = image.unsqueeze(1)
        B, T, C, H, W = image.size()

        if self.use_backbone:
            backbone_feature = self.backbone(image.flatten(end_dim=1))
        
        # savi
        if self.use_backbone:
            emb = self.steve_encoder.cnn_backbone(backbone_feature)      # B * T, cnn_hidden_size, H, W
        else:
            emb = self.steve_encoder.cnn(image.flatten(end_dim=1))
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)         # slots: B, T, num_slots, slot_size
                                                                # attns: B, T, num_slots, num_inputs
        return slots[:,0,:,:], attns[:,0,:,:]  # return the slots of the first frame, shape: B, num_slots, slot_size
        