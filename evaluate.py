import os
import sys
import json
import pyter
import numpy as np
from tqdm.auto import tqdm
sys.path.insert(0, "./NMTEvaluation/")
sys.path.insert(0, "./NMTEvaluation/metrics/")
from nmt_eval import compute_metrics


from utils import *

# Do not assign GPUs
os.environ['CUDA_VISIBLE_DEVICES']=''

def main():
    # Load configuration arguments
    args = read_args(sys.argv[1])

    # Cache folder for the metrics
    if not os.path.exists(args.root_dir + args.cache_dir):
        os.makedirs(args.root_dir + args.cache_dir)

    for lang in args.languages:
        path = args.root_dir + args.output_dir + args.predictions_dir + \
            lang + '/' + 'generated_text_{}_{}.txt'.format(
                args.model_type, lang
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                'The file you try to open, {}, does not exist. Have you run ' \
                'the "translate.py" script?'.format(path)
            )

        metrics = load_metrics(args)

        hyps, refs = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    hyp, ref = line.rstrip().split('\t')
                except IOError:
                    print(
                        'Something wrong occurred when reading the line\n' \
                        '{}.'.format(repr(line))
                    )
                    raise
                hyps.append(hyp)
                refs.append(ref)
                hyp, ref = postprocess_text([hyp], [ref])
                for metric in args.metrics:
                    metrics[metric.lower()].add_batch(
                        predictions=hyp,
                        references=ref
                    )
        print('-------------------')
        print(lang)
        print('-------------------')
        scores = compute_metrics(refs, hyps)
        for k,v in scores.items():
            print('{}: {:.2f}'.format(k,v))
        # Save results
        path = args.root_dir + args.output_dir + args.predictions_dir + \
            lang + '/' + 'metrics_{}_{}.txt'.format(
                args.model_type, lang
            )
        save_metrics = args.always_save_metrics     or not os.path.exists(path)
        if os.path.exists(path):
            if not args.always_save_metrics:  
                repeat = True
                while repeat:
                    print('A file {} already exists, would you like to ' \
                        'overwrite it? [Y/N]'    
                    )
                    res = input()
                    if res.lower() == 'y':
                        save_metrics = True
                        repeat = False
                    elif res.lower() == 'n':
                        save_metrics = False
                        repeat = False
        s = '-'*10
        s += '\n'
        s += 'Language: {}\n'.format(lang)
        s += '-'*10
        s += '\n'
        #s += 'TER: {:.2f}\n'.format(np.mean(ter))
        
        for metric in args.metrics:
            eval_score = metrics[metric.lower()].compute()
            s += '{}: {:.2f}\n'.format(metric, eval_score['score'])
        if save_metrics:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(s)
        else:
            print(s)

if __name__ == '__main__':
    main() 