import logging

from ..tresnet_v2 import TResnetL_V2

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes,'remove_aa_jit': args.remove_aa_jit}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name=='tresnet_l_v2':
        model = TResnetL_V2(model_params)
    elif args.model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    return model
