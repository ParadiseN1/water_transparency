# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import sys

sys.path.append('/app/SimVPv2')
import warnings
warnings.filterwarnings('ignore')

from simvp.api import NonDistExperiment
from simvp.utils import create_parser, load_config, update_config
import wandb
try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    wandb.init(project="simvp2_secchi_depth", name=args.ex_name)
    # wandb.init(project="simvp2_secchi_depth", name="test MMNIST")
    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    config = update_config(config, load_config(cfg_path),
                           exclude_keys=['batch_size', 'val_batch_size', 'sched'])

    exp = NonDistExperiment(args)
    print('>'*35 + ' training ' + '<'*35)
    exp.train()

    print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()
    if has_nni:
        nni.report_final_result(mse)
