import copy

import torch.nn as nn
import torch
import math

from experiments.models.tst.ts_transformer import TSTransformerEncoder as _TSEncoder

""" Architectures for time series regression tasks.

	Conventions:
	- Input shape: (batch_size, seq_length, feat_dim)
	- Output shape: (batch_size, output_size)
	- Each architecture class should accept feat_dim as a parameter.

"""


# ---------------------------------------------------------------------------
# F2F
# https://openreview.net/forum?id=9aElHWiZ72
# ---------------------------------------------------------------------------

class _F2FPretrainHead(nn.Module):
	"""Multi-head projector for self-supervised pretraining."""

	def __init__(self, input_size, hidden_layer, dropout, output_size, num_heads=1):
		super().__init__()
		mlp = nn.Sequential(
			nn.Linear(input_size, hidden_layer),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(hidden_layer, output_size),
		)
		self.heads = nn.ModuleList([copy.deepcopy(mlp) for _ in range(num_heads)])

	def forward(self, x):
		return [head(x) for head in self.heads]



class F2F(nn.Module):
	"""Self-supervised pretrained encoder + regression head.

	Uses TSTransformerEncoder as backbone.  During both pretraining and
	fine-tuning, _backbone_embed extracts (B, T, d_model) embeddings
	(skipping backbone.output_layer).  The head maps the flattened
	embeddings to the target.

	The two-phase training (pretrain then fine-tune) is handled by run_f2f_experiments.
	"""

	def __init__(self, feat_dim, seq_len, d_model, n_heads, num_layers,
				 dim_feedforward=256, output_size=1, dropout=0.1,
				 pos_encoding='fixed', activation='gelu', norm='BatchNorm',
				 head_hidden=512, head_dropout=0.0):
		super().__init__()
		self.seq_len = seq_len
		self.d_model = d_model

		self.backbone = _TSEncoder(
			feat_dim=feat_dim,
			max_len=seq_len,
			d_model=d_model,
			n_heads=n_heads,
			num_layers=num_layers,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			pos_encoding=pos_encoding,
			activation=activation,
			norm=norm,
			freeze=False,
		)

		embed_dim = d_model * seq_len
		self.head = nn.Sequential(
			nn.Linear(embed_dim, head_hidden),
			nn.ReLU(inplace=True),
			nn.Dropout(head_dropout),
			nn.Linear(head_hidden, output_size),
		)

	def _backbone_embed(self, x):
		"""Run backbone encoder and return intermediate (B, T, d_model) embeddings.

		Replicates the TSTransformerEncoder forward pass but stops before
		output_layer, matching TSTransformerEncoderClassiregressor behaviour.
		"""
		bb = self.backbone
		padding_masks = torch.ones(x.shape[0], x.shape[1],
								   dtype=torch.bool, device=x.device)
		inp = x.permute(1, 0, 2)
		inp = bb.project_inp(inp) * math.sqrt(bb.d_model)
		inp = bb.pos_enc(inp)
		output = bb.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
		output = bb.act(output)
		output = output.permute(1, 0, 2)
		output = bb.dropout1(output)
		return output  # (B, T, d_model)

	def forward(self, x):
		emb = self._backbone_embed(x)                      # (B, T, d_model)
		emb = emb.reshape(emb.shape[0], -1)                # (B, T * d_model)
		out = self.head(emb)                                # (B, output_size)
		return out[:, -1]


# ---------------------------------------------------------------------------
# ModernTCN
# https://openreview.net/forum?id=vpJMJerXHU
# ---------------------------------------------------------------------------

from experiments.models.modern_tcn.ModernTCN import ModernTCN as _OrigModernTCN


class ModernTCN(nn.Module):
	def __init__(self, feat_dim, seq_len, output_size=1, d_model=32,
				 num_blocks=2, large_kernel=51, small_kernel=5,
				 patch_size=1, patch_stride=1, ffn_ratio=1,
				 dropout=0.1, head_dropout=0.1):
		super().__init__()
		self.model = _OrigModernTCN(
			task_name='regression',
			patch_size=patch_size,
			patch_stride=patch_stride,
			stem_ratio=1,
			downsample_ratio=1,
			ffn_ratio=ffn_ratio,
			num_blocks=[num_blocks],
			large_size=[large_kernel],
			small_size=[small_kernel],
			dims=[d_model],
			dw_dims=[d_model],
			nvars=feat_dim,
			small_kernel_merged=False,
			backbone_dropout=dropout,
			head_dropout=head_dropout,
			use_multi_scale=True,
			revin=False,
			affine=False,
			subtract_last=False,
			freq=None,
			seq_len=seq_len,
			c_in=(feat_dim,),
			individual=False,
			target_window=1,
			class_drop=head_dropout,
			class_num=1,
			output_size=output_size,
		)

	def forward(self, x):
		# x: (B, T, C) — repo convention → (B, C, T) for original ModernTCN
		x = x.transpose(1, 2)
		x = self.model(x)
		return x[:, -1]


# ---------------------------------------------------------------------------
# xLSTM-Mixer
# https://arxiv.org/abs/2410.16928
# ---------------------------------------------------------------------------

from experiments.models.xlstm_mixer.xlstm_mixer import xLSTMMixer as _OrigxLSTMMixer, AblationMode


class xLSTMMixer(nn.Module):
	def __init__(self, feat_dim, seq_len, output_size=1,
				 xlstm_embedding_dim=256, xlstm_num_heads=8,
				 xlstm_num_blocks=1, xlstm_dropout=0.1,
				 xlstm_conv1d_kernel_size=0, num_mem_tokens=4,
				 backcast=True):
		super().__init__()
		self.model = _OrigxLSTMMixer(
			pred_len=0,
			seq_len=seq_len,
			enc_in=feat_dim,
			xlstm_embedding_dim=xlstm_embedding_dim,
			num_mem_tokens=num_mem_tokens,
			num_tokens_per_variate=1,
			xlstm_dropout=xlstm_dropout,
			xlstm_conv1d_kernel_size=xlstm_conv1d_kernel_size,
			xlstm_num_heads=xlstm_num_heads,
			xlstm_num_blocks=xlstm_num_blocks,
			backcast=backcast,
			packing=1,
			ablation_mode=AblationMode.SLSTM_MEMORY_BACK,
		)

	def forward(self, x):
		return self.model.classification(x)[:, -1]                 # (B, 1) → (B,)


# ---------------------------------------------------------------------------
# TST
# https://arxiv.org/abs/2010.02803
# ---------------------------------------------------------------------------

from experiments.models.tst.ts_transformer import TSTransformerEncoderClassiregressor as _OrigTST


class TST(nn.Module):
	def __init__(self, feat_dim, seq_len, output_size=1, d_model=128, n_heads=8,
				 num_layers=4, dim_feedforward=256, dropout=0.1,
				 pos_encoding='fixed', activation='gelu', norm='BatchNorm'):
		super().__init__()
		self.model = _OrigTST(
			feat_dim=feat_dim,
			max_len=seq_len,
			d_model=d_model,
			n_heads=n_heads,
			num_layers=num_layers,
			dim_feedforward=dim_feedforward,
			num_classes=output_size,
			dropout=dropout,
			pos_encoding=pos_encoding,
			activation=activation,
			norm=norm,
			freeze=False,
		)

	def forward(self, x):
		# x: (B, T, C) — repo convention
		# Original expects padding_masks: (B, T) boolean, True = keep
		padding_masks = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
		out = self.model(x, padding_masks)  # (B, output_size)
		return out[:, -1]


# ---------------------------------------------------------------------------
# PatchTSMixer
# https://arxiv.org/abs/2306.09364
# ---------------------------------------------------------------------------

class PatchTSMixer(nn.Module):
	"""Wrapper around ``transformers.PatchTSMixerForRegression``.

	Uses the full HF model including its built-in regression head.
	Output: ``regression_outputs`` of shape ``(batch_size, num_targets)``.
	Requires ``pip install transformers``.
	"""

	def __init__(self, feat_dim, seq_len, output_size=1, d_model=64,
				 num_layers=8, patch_length=8, expansion_factor=2,
				 dropout=0.1, head_dropout=0.1):
		super().__init__()
		from transformers import PatchTSMixerConfig
		from transformers import PatchTSMixerForRegression as _HFModel

		config = PatchTSMixerConfig(
			context_length=seq_len,
			num_input_channels=feat_dim,
			patch_length=patch_length,
			d_model=d_model,
			num_layers=num_layers,
			expansion_factor=expansion_factor,
			dropout=dropout,
			head_dropout=head_dropout,
			num_targets=output_size,
			head_aggregation="use_last",
			mode="common_channel",
		)
		self.model = _HFModel(config)

	def forward(self, x):
		# x: (B, T, C)
		out = self.model(past_values=x)
		# regression_outputs: (B, num_targets)
		return out.regression_outputs[:, -1]