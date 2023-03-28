from yacs.config import CfgNode as CN

_C = CN()

# Experiment
_C.exp = CN()
_C.exp.id = -1
_C.exp.seed = 1
_C.exp.epochs = 100
_C.exp.batch_size = 32
_C.exp.test_batch_size = 64
_C.exp.log_interval = 1000

# Model
_C.model = CN()
_C.model.dropout = 0.0
_C.model.loss = 'bce_loss'
_C.model.k = 20

# Data
_C.data = CN()
_C.data.name = 'breaking_bad'
_C.data.prop_dt = 0.1
_C.data.prop_test = 0.7
_C.data.prop_val = 0.7
_C.data.prop_pn = 0.75

# Optimizer
_C.opt = CN()
_C.opt.scheduler = 'StepLR'
_C.opt.name = 'Adadelta'
_C.opt.lr = 1.0
_C.opt.gamma = 0.9
_C.opt.weight_decay = 1e-4

def get_cfg_defaults():
    return _C.clone()
