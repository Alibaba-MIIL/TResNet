import torch
from src.helper_functions.helper_functions import validate, create_dataloader
from src.models import create_model
import argparse

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch TResNet ImageNet Inference')
parser.add_argument('--val_dir')
parser.add_argument('--model_path')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--remove_aa_jit', action='store_true', default=False)


def main():
    # parsing args
    args = parser.parse_args()

    # setup model
    print('creating model...')
    model = create_model(args).cuda()
    state = torch.load(args.model_path, map_location='cpu')['model']
    model.load_state_dict(state, strict=False)
    model.eval()
    print('done\n')

    # setup data loader
    print('creating data loader...')
    val_loader = create_dataloader(args)
    print('done\n')

    # actual validation process
    print('doing validation...')
    prec1_f = validate(model, val_loader)
    print("final top-1 validation accuracy: {:.2f}".format(prec1_f.avg))


if __name__ == '__main__':
    main()
