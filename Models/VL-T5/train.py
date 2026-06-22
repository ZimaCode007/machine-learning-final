"""Single-GPU training launcher for VL-T5 on Windows."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vqa import main_worker
from param import parse_args
import torch

if __name__ == "__main__":
    args = parse_args()
    args.gpu = 0
    args.rank = 0
    args.world_size = 1
    args.distributed = False
    args.multiGPU = False

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')
    comments = []
    if args.load is not None:
        ckpt_str = "_".join(args.load.split('/')[-3:])
        comments.append(ckpt_str)
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)
    run_name = f'{current_time}_GPU1'
    if comments:
        run_name += f'_{comment}'
    args.run_name = run_name

    print(args)
    main_worker(0, args)
