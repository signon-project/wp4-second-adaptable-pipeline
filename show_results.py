import sys
import json

from utils import read_args

def main():
    # Load configuration arguments
    args = read_args(sys.argv[1])

    subpath = args.root_dir + args.output_dir + args.predictions_dir
    for lang in args.languages:
        metrics_file = subpath + '{}/metrics_{}_{}.txt'.format(
            lang, args.model_type, lang
        )
        print('-'*10)
        print('{} test results'.format(lang))
        print('-'*10)
        with open(metrics_file, 'r') as f:
            i = 0
            for line in f:
                if i < 3: 
                    i += 1
                    continue
                
                print(line.rstrip())


if __name__ == "__main__":
    main()