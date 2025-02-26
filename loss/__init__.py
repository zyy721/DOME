from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .ce_loss import CeLoss
from .emb_loss import VQVAEEmbedLoss
from .recon_loss import ReconLoss, LovaszLoss