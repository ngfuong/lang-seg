from argparse import ArgumentParser
from modules.trans_lseg_module_zs import TransLSegModuleZS 
from utils import do_training, do_training_debug, get_default_argument_parser

if __name__ == "__main__":
    parser = TransLSegModuleZS.add_model_specific_args(get_default_argument_parser())
    debug_parser = ArgumentParser(parents=[parser], add_help=False)
    debug_parser.add_argument(
        "--debug",
        action='store_true', # Auto set --debug=False when argument is not present
    )
    args = debug_parser.parse_args()
    print('=== args ===')
    print(args)
    print('============')
    if args.debug:
        print("************DEBUGGING***********")
        do_training_debug(args, TransLSegModuleZS, shot="few_shot")
    else:
        do_training(args, TransLSegModuleZS, shot="few_shot")