import os
import sys
import json
import sentencepiece as spm
from shutil import copyfile, rmtree

from utils import jsonParser

def main():
    # Load configuration arguments
    try:
        with open(sys.argv[1]) as f:
            args = json.loads(f.read(), object_hook=jsonParser)
    except:
        raise FileNotFoundError('ERROR: Provide a JSON with arguments as ' + \
            ' in "python train_tokeniser.py args.json"'
        )

    if os.path.exists(args.root_dir + args.tokeniser_dir):
        rmtree(args.root_dir + args.tokeniser_dir)
    os.makedirs(args.root_dir + args.tokeniser_dir)

    if not os.path.exists(args.root_dir + args.all_corpus):
        raise FileNotFoundError(
            'The file {} does not exist and is required by the ' \
            '`train_tokeniser.py` script.'.format(
                args.root_dir + args.all_corpus
            )
        )
    
    # See https://github.com/google/sentencepiece/blob/master/doc/options.md
    # for all the available options. `pad_id` must be set
    spm.SentencePieceTrainer.train(
        input=args.root_dir + args.all_corpus,
        model_prefix=args.root_dir + args.tokeniser_dir + args.tokeniser_prefix,
        vocab_size=args.vocabulary_size,
        character_coverage=args.tokeniser_character_coverage, 
        shuffle_input_sentence=True,
        input_sentence_size=args.tokeniser_input_sentence_size,
        seed_sentencepiece_size=args.seed,
        unk_piece=args.unk_token, 
        bos_piece=args.bos_token, 
        eos_piece=args.eos_token, 
        pad_piece=args.pad_token,
        pad_id=3 # being 0, 1 and 2 the unk, bos and eos tokens, respectively
    )

if __name__ == "__main__":
    main() 