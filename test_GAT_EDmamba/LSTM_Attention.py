from .GAT import SensorSpatialGAT
from mamba_ssm import Mamba

import torch
from torch import nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 mamba_num_layers,
                 mamba_d_model,
                 mamba_d_state=16,
                 mamba_dropout=0.1,
                 mamba_d_conv=4,
                 mamba_expand=2):
        super(Seq2SeqEncoder, self).__init__()
        self.input_size = int(input_size)
        self.mamba_d_model = int(mamba_d_model)
        self.mamba_num_layers = int(mamba_num_layers)
        self.output_size = self.mamba_d_model
        self.hidden_size = self.mamba_d_model
        self.input_projection = None
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=self.mamba_d_model,
                d_state=mamba_d_state,
                dropout=mamba_dropout,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
            for _ in range(self.mamba_num_layers)
        ])

    def forward(self, x):
        if self.input_projection is None:
            input_size = int(x.size(-1))
            self.input_projection = (
                nn.Identity()
                if input_size == self.mamba_d_model
                else nn.Linear(input_size, self.mamba_d_model)
            ).to(x.device)
        x = self.input_projection(x)
        output = x
        for block in self.blocks:
            output = block(output)
        hidden_state = output[:, -1, :].unsqueeze(0).repeat(self.mamba_num_layers, 1, 1)
        cell_state = torch.zeros_like(hidden_state)
        return output, (hidden_state, cell_state)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class Seq2SeqDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 mamba_num_layers,
                 mamba_d_model,
                 use_aef=False,
                 mamba_d_state=16,
                 mamba_dropout=0.1,
                 mamba_d_conv=4,
                 mamba_expand=2):
        super(Seq2SeqDecoder, self).__init__()
        self.input_size = int(input_size)
        self.mamba_d_model = int(mamba_d_model)
        self.mamba_num_layers = int(mamba_num_layers)
        self.use_aef = bool(use_aef)
        self.input_projection = (
            nn.Identity()
            if self.input_size == self.mamba_d_model
            else nn.Linear(self.input_size, self.mamba_d_model)
        )
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=mamba_d_model,
                d_state=mamba_d_state,
                dropout=mamba_dropout,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
            for _ in range(self.mamba_num_layers)
        ])
        self.Linear = nn.Linear(self.mamba_d_model, 1)

    def forward(self, encoder_output, encoder_state):
        decoder_input = self.input_projection(encoder_output)
        output = decoder_input
        for block in self.blocks:
            output = block(output)
        output = self.Linear(output[:, -1, :])
        return output, None


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 use_spatial_gat=True,
                 graph_mode='dynamic_knn',
                 num_sensors=14,
                 gat_hidden_dim=8,
                 gat_out_dim=16,
                 gat_num_layers=2,
                 gat_embed_dim=16,
                 gat_topk=5,
                 dropout=0.1,
                 gat_alpha=0.1,
                 use_decoder=True):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_spatial_gat = use_spatial_gat
        self.graph_mode = graph_mode
        self.use_decoder = bool(use_decoder)

        if self.use_decoder and self.decoder is None:
            raise ValueError("decoder cannot be None when use_decoder=True")

        if use_spatial_gat:
            self.spatial_gat = SensorSpatialGAT(
                num_sensors=num_sensors,
                in_features=1,
                hidden_features=gat_hidden_dim,
                out_features=gat_out_dim,
                num_layers=gat_num_layers,
                embed_dim=gat_embed_dim,
                topk=gat_topk,
                graph_mode=graph_mode,
                dropout=dropout,
                alpha=gat_alpha,
            )
        else:
            self.spatial_gat = None

        encoder_hidden_size = int(
            getattr(self.encoder, 'output_size', 0)
            or getattr(self.encoder, 'hidden_size', 0)
        )
        if encoder_hidden_size <= 0:
            raise ValueError("failed to infer encoder hidden size")
        self.encoder_hidden_size = encoder_hidden_size

        if not self.use_decoder:
            self.encoder_regressor = nn.Linear(self.encoder_hidden_size, 1)
        else:
            self.encoder_regressor = None

    def forward(self, encoder_x):
        use_decoder = bool(getattr(self, 'use_decoder', True))

        if self.use_spatial_gat:
            encoder_x = self.spatial_gat(encoder_x)

        encoder_output, encoder_state = self.encoder(encoder_x)

        if use_decoder:
            output, attention_weights = self.decoder(encoder_output, encoder_state)
            return output, attention_weights

        last_hidden = encoder_output[:, -1, :]
        output = self.encoder_regressor(last_hidden)
        attention_weights = torch.zeros(
            encoder_output.size(0),
            encoder_output.size(1),
            device=encoder_output.device,
            dtype=encoder_output.dtype,
        )
        return output, attention_weights
