from modules.trans_lseg_module_zs import TransLSegModuleZS
from utils import do_training, get_default_argument_parser

if __name__ == "__main__":
    parser = TransLSegModuleZS.add_model_specific_args(get_default_argument_parser())
    args = parser.parse_args()
    print('=== args ===')
    print(args)
    print('============')
    do_training(args, TransLSegModuleZS, shot="few_shot")
