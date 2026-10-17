"""Evaluate four-view checkpoints on held-out subjects 0500..0525."""
import argparse
import copy
import json
from pathlib import Path

from safetensors.torch import load_file
from core.options import config_defaults
from core.THuman_supervised_4_12 import THuman
from main_thuman_cycle_4views import FourViewLGM, evaluate
from train_thuman_cycle_4views import subject_split, preflight, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='THumanDataset/thuman2_rendered_lgm')
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()
    output = Path(args.workspace)
    if (output / 'evaluation').exists():
        raise FileExistsError('Use a new evaluation workspace')
    output.mkdir(parents=True, exist_ok=True)
    opt = copy.deepcopy(config_defaults['big'])
    opt.output_size = 256
    dataset = THuman(opt, split='test', training=False)
    dataset.root = str(Path(args.data_root).resolve())
    _, dataset.items = subject_split()
    preflight(Path(dataset.root), dataset.items)
    model = FourViewLGM(opt).cuda()
    missing = model.load_state_dict(load_file(args.checkpoint), strict=False)
    if missing.unexpected_keys or any(not k.startswith('lpips_loss.') for k in missing.missing_keys):
        raise RuntimeError(f'Checkpoint mismatch: {missing}')
    records = evaluate(model, [('test', dataset, len(dataset))], output, 'evaluation')
    metrics = aggregate(records)['test']
    (output / 'summary.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
