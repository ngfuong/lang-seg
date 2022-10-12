from modules.lseg_module_zs import LSegModuleZS
from utils import do_training, get_default_argument_parser

if __name__ == "__main__":
    parser = LSegModuleZS.add_model_specific_args(get_default_argument_parser())
    args = parser.parse_args()
    # shot="few_shot" means train zero shot
    print('=== args ===')
    print(args)
    print('============')
    do_training(args, LSegModuleZS, shot="few_shot")