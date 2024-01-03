# python add_languages.py args.json --langs=en-ga,ga-en 

import os
# Do not assign GPUs
os.environ['CUDA_VISIBLE_DEVICES']=''

import json
import torch
import argparse
import sentencepiece as spm
from tqdm import tqdm
from typing import List, Dict
from shutil import move, rmtree
from torch.nn import Embedding, Linear
from transformers import MBartForConditionalGeneration

from utils import *

def main():
    # For reading arguments
    parser = argparse.ArgumentParser(
        description='Add new languages to an existing mBART model.'
    )
    parser.add_argument(
        '--file', metavar='file path', type=str,
        help='a path to a file with comma-separated language pairs'
    )
    parser.add_argument(
        '--langs', metavar='lang pair', type=str,
        help='list of language pairs (<src_lang>-<tgt_lang>)'
    )
    parser.add_argument(
        "--from_saved", action='store_true',
        help="Whether a saved mBART model should be modified",
    )
    parser.add_argument(
        "--random_new_embeddings", action='store_true',
        help="Whether a finetuned mBART model should be modified",
    )
    sys_args, args_file = parser.parse_known_args()
    
    if sys_args.langs == None and sys_args.file == None:
        raise ValueError('At least one language or a path to a file ' \
            'containing various languages (comma separated) is expected.')

    # Load configuration arguments
    try:
        args_file = args_file[0]
        with open(args_file) as f:
            args = json.loads(f.read(), object_hook=jsonParser)
    except:
        raise FileNotFoundError(
            'Provide a JSON with arguments as in "python add_languages.py ' \
            'args.json"'
        )

    # Take the list of languages from arguments
    if sys_args.file == None:
        new_languages = sys_args.langs.split(',')
    # Read new languages from a file
    else:
        try:
            with open(sys_args.file, 'r') as f:
                new_languages = f.readline().rstrip().split(',')
        except:
            raise FileNotFoundError(
                'The file {} does not exist.'.format(sys_args.file)
            )
    # Remove undesired characters
    for i in range(len(new_languages)):
        new_languages[i] = new_languages[i].rstrip().lstrip()

    # Check if all the new languages are in mBART, otherwise add the
    # new language tokens
    new_language_codes = []
    new_langs = {}
    for lang_pair in new_languages:
        for lang in lang_pair.split('-'):
            new_langs[lang] = True

    for lang in new_langs.keys():
        for lang_code in args.language_codes:
            if lang in lang_code: 
                new_langs[lang] = False
    
    for lang in new_langs.keys():
        if new_langs[lang]:
            new_language_codes.append(
                '{}_{}'.format(lang.lower(), lang.upper())
            )
    # The array `new_language_codes` contains all the new languages that must
    # be added to the tokeniser, creating their own token

    if len(new_language_codes) == 0:
        raise ValueError(
            'No new languages included.'
        )
    
    # Append to the "all corpus" file the new languages' samples
    open(args.root_dir + args.all_corpus, 'w').close() # to clean the content
    for lang in args.languages + new_languages:
        print('Adding the {} corpus'.format(lang))
        path = args.root_dir + args.corpora_dir + lang + \
            '/train_{}.txt'.format(lang)
        if not os.path.exists(path):
            raise FileNotFoundError('The file {} does not exist, cannot ' \
                'train a new tokeniser without it.'.file(path))
        with open(path, 'r') as f:
            if not os.path.exists(args.root_dir + args.all_corpus):
                raise FileNotFoundError('The file {} does not exist.'.format(
                    args.root_dir + args.all_corpus
                ))
            with open(args.root_dir + args.all_corpus, 'a') as allcorpus:
                i = 0
                for line in tqdm(f, total=args.max_samples_per_lang):
                    allcorpus.write(line)
                    i += 1
                    # Sample `args.max_samples_per_lang` elements from each
                    # language-pair
                    if i == args.max_samples_per_lang:
                        break
    
    # Save previous tokeniser (if any)
    if os.path.exists(args.root_dir + args.tokeniser_dir):     
        if not os.path.exists(args.root_dir + args.old_tokeniser_dir):
            os.makedirs(args.root_dir + args.old_tokeniser_dir)
        src_path = args.root_dir + args.tokeniser_dir
        tgt_path = args.root_dir + args.old_tokeniser_dir
        rmtree(args.root_dir + args.old_tokeniser_dir)
        # Move files to the old tokeniser folder
        move(src_path, tgt_path)
    os.makedirs(args.root_dir + args.tokeniser_dir)

    # Train the tokeniser with the new languages' corpus included
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
        pad_id=3, # being 0, 1 and 2 the unk, bos and eos tokens, respectively
        control_symbols=','.join(new_language_codes)
    )

    language_codes = args.language_codes + new_language_codes
    
    # Load the model to modify and the original dictionary
    # Can specify whether the original pretrained mBART model is loaded
    # or the last model saved (pruned vocabulary or not)
    if not sys_args.from_saved:
        vocab_path = args.root_dir + args.pruned_model + \
            args.pruned_vocabulary_file
        if not os.path.exists(vocab_path):
            raise FileNotFoundError('The file {} was not found.'.format(
                vocab_path
            ))
        pre_dict = load_dict(
            language_codes, vocab_path
        )
        model = MBartForConditionalGeneration.from_pretrained(
            args.root_dir + args.pruned_model
        )
    else:
        """ if args.pruned_vocabulary:
            print('AQUI SI')
            vocab_path = args.pruned_model + args.pruned_vocabulary_file
        else:
            print('aqui no!!!')
            vocab_path = args.original_pruned_vocabulary_file """
        vocab_path = args.pruned_model + args.pruned_vocabulary_file
        if not os.path.exists(args.root_dir + vocab_path):
            raise FileNotFoundError('The file {} was not found.'.format(
                args.root_dir + vocab_path
            ))
        print(args.root_dir + vocab_path)
        pre_dict = load_dict(
            language_codes, args.root_dir + vocab_path
        )
        model = torch.load(
            args.root_dir + args.output_dir + \
            args.saved_models_dir + args.last_model_dir + 'model.pt',
            map_location ='cpu'
        )
        print('no',model.config)
    
    # Save previous pruned vocabulary file
    path = args.root_dir + args.pruned_model + args.pruned_vocabulary_file
    move(
        path,
        args.root_dir + args.pruned_model +
            args.old_pruned_vocabulary_file
    )
        
    # Generate new vocabulary with the new languages added
    os.system(
        'spm_encode --model={} --generate_vocabulary < {} > {}'.format(
            args.root_dir + args.tokeniser_dir + args.tokeniser_prefix + \
                '.model',
            args.root_dir + args.all_corpus,
            args.root_dir + args.pruned_model + args.pruned_vocabulary_file
        ))
    
    # Remove the \t character and replace it by a blank space
    os.system("sed -ie 's/\t/ /g' {}".format(
        args.root_dir + args.pruned_model + args.pruned_vocabulary_file)
    )

    mapping, ft_dict, new_vocab = mapping_of_vocabularies(
        args,
        args.root_dir + args.pruned_model + args.old_pruned_vocabulary_file,
        args.root_dir + args.pruned_model + args.pruned_vocabulary_file,
        new_language_codes = new_language_codes
    )
    
    model = assign_embeddings_from_mapping(
        args,
        model,
        mapping,
        ft_dict,
        new_vocab
    )

    # Save the model using the HuggingFace format and also the Pytorch standard
    torch.save(model, args.root_dir + args.pruned_model + 'model.pt')
    model.save_pretrained(args.root_dir + args.pruned_model)

if __name__ == '__main__':
    main() 