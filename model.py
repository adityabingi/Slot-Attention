import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class SlotAttention(nn.Module):

    def __init__(self, in_dim, slot_size, num_slots, num_iters, mlp_hdim, 
                                                                epsilon =1e-8, implicit_grads=False):

        super().__init__()

        self.num_slots = num_slots
        self.num_iters = num_iters
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.implicit_grads = implicit_grads

        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(in_dim, slot_size, bias=False)
        self.project_v = nn.Linear(in_dim, slot_size, bias=False)

        self.norm_inputs = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)


        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hdim),
            nn.ReLU(),
            nn.Linear(mlp_hdim, slot_size)
            )

        self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.slot_size)))
        self.slots_logsigma = nn.Parameter(nn.init.xavier_uniform_(torch.ones( 1, self.slot_size)))


    def step(self, slots, k, v):

        slots_prev = slots
        slots = self.norm_slots(slots)
        q = self.project_q(slots) # shape: [batch_size, num_slots, slot_size]
        scores = (self.slot_size ** -0.5) * torch.matmul(k, q.transpose(2, 1))
        attn = torch.softmax(scores, dim=-1) # shape: [batch_size, num_inputs, num_slots]

        #weighted mean 
        attn = attn + self.epsilon
        attn = attn/ torch.sum(attn, dim=1, keepdim=True) #shape: [batch_size, num_inputs, num_slots]

        updates = torch.matmul(attn.transpose(2, 1), v) #shape: [batch_size, num_slots, slot_size]

        slots = self.gru(updates.reshape(-1, self.slot_size), slots_prev.reshape(-1, self.slot_size))
        slots = slots.reshape(batch_size, self.num_slots, self.slot_size)
        slots = self.norm_mlp(slots)
        #slots = self.mlp(slots)
        slots = slots + self.mlp(slots)

        return slots

    def forward(self, x):

        batch_size, num_inputs, in_dim = x.shape
        x = self.norm_inputs(x)
        k = self.project_k(x) # shape:[batch_size, num_inputs, slot_size]
        v = self.project_v(x) # shape:[batch_size, num_inputs, slot_size]

        mu = self.slots_mu.repeat(batch_size, self.num_slots, 1)
        logsigma = self.slots_logsigma.repeat(batch_size, self.num_slots, 1)
        logsigma = F.softplus(logsigma) + 1e-5
        slots_dist = dist.independent.Independent(dist.Normal(loc=mu, scale=logsigma), 1)
        slots = slots_dist.rsample()

        for _ in range(self.num_iters):
            slots = self.step(slots, k, v)

        if self.implicit_grads:
            slots = self.step(slots.detach(), k, v)

        return slots

class SlotAttentionModel(nn.Module):

    def __init__(self, 
           resolution, 
           num_slots,
           num_iters,
           device,
           in_channels =3, 
           num_hidden = 4,
           hdim = 32,
           slot_size = 64,
           slot_mlp_size = 128,
           decoder_resolution=(35, 35)):

        super().__init__()

        self.resolution = resolution
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.decoder_resolution = decoder_resolution

        modules = []
        in_dim = self.in_channels
        for _ in range(num_hidden):
            modules.append(nn.Conv2d(in_dim, hdim, kernel_size=5, stride=1, padding=5//2))
            modules.append(nn.ReLU())
            in_dim = hdim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embed = PositionEmbed(hdim, self.resolution, device)

        self.norm_layer = nn.LayerNorm(hdim)
        self.pre_slot_encode = nn.Sequential(
                                    nn.Linear(hdim, hdim),
                                    nn.ReLU(),
                                    nn.Linear(hdim, hdim)
                                )

        self.slot_attention = SlotAttention(
                                in_dim = hdim,
                                slot_size=self.slot_size,
                                num_slots=self.num_slots,
                                num_iters=num_iters,
                                mlp_hdim =slot_mlp_size
                              )
        
        self.decoder_pos_embed = PositionEmbed(slot_size, self.decoder_resolution, device)

        modules = []
        in_dim = slot_size
        for _ in range(num_hidden-1):
            modules.append(nn.ConvTranspose2d(in_dim, hdim, kernel_size=5, stride=1, padding=5//2))
            modules.append(nn.ReLU())
            in_dim = hdim

        modules.append(nn.ConvTranspose2d(32, 4, kernel_size=3, stride=1, padding=3//2))

        self.decoder = nn.Sequential(*modules)

    def forward(self, x):

        batch_size, num_channels, height, width = x.shape

        x = self.encoder(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.encoder_pos_embed(x) #shape:[batch_size, height, width, hdim]

        x = torch.flatten(x, start_dim=1, end_dim=2) #shape:[batch_size, height*width, hdim]

        x = self.norm_layer(x) 
        x = self.pre_slot_encode(x) 

        slots = self.slot_attention(x) #shape:[batch_size, num_slots, slot_size]

        x = slots.reshape(-1,1,1, self.slot_size).repeat(1, *self.decoder_resolution, 1)
        # x has shape: [batch_size*num_slots, decoder_res[0], decoder_res[1], slot_size]

        x = self.decoder_pos_embed(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.decoder(x) 
        #x has shape:[batch_size*num_slots, num_channels+1, height, width]

        x = x.reshape(-1, self.num_slots, num_channels+1, height, width)

        recon, masks = x.split((3, 1), dim=2)
        masks = F.softmax(masks, dim=1)
        recon_combined = (recon * masks).sum(dim=1)

        return recon_combined, recon, masks, slots

class PositionEmbed(nn.Module):

    def __init__(self, hdim, resolution, device):
        super().__init__()

        self.dense = nn.Linear(4, hdim)
        self.grid = build_grid(resolution).to(device)

    def forward(self, x):

        grid = self.dense(self.grid)
        return x + grid

def build_grid(resolution):

    grid = torch.meshgrid(*[torch.linspace(0.0, 1.0, r) for r in resolution])
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    grid = torch.cat([grid, 1.0-grid], dim=-1)
    return grid
