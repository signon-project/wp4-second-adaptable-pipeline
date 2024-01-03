import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Add new languages to an existing mBART model.'
    )
    parser.add_argument(
        "--mode",
        help="Whether to train a text2text model or text2amr model."
    )
    sys_args, args_file = parser.parse_known_args()

    if isinstance(sys_args, tuple):
        sys_args = sys_args[0]
    
    if sys_args.mode is None:
        raise ValueError(
            'The `mode` argument must be set to either ' +
            '`text2text` or `text2amr`.'
        )
    mode = sys_args.mode
    if len(args_file) == 0:
        raise IOError('If the `text2text` is used, a configuration ' +
        'file must be provided.')
    if isinstance(args_file, list):
        args_file = args_file[0]

    if mode == 'text2text':
        
        from finetune import main as text2text
        text2text(args_file)
    elif mode == 'text2amr':
        """ from finetune import main as text2amr
        text2amr() """
        path = '../multilingual-text-to-amr/'
        sys.path.append(path)
        os.system("run-mbart-amr {}".format(path + args_file))
    else:
        raise ValueError(
            'The `mode` argument must be set to either ' +
            '`text2text` or `text2amr`.'
        )

if __name__ == "__main__":
    main()