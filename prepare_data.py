# python prepare_data.py args.json --pretrained_mbart

import os
import sys
import json
import urllib
import random
import argparse
import sentencepiece as spm
from tqdm import tqdm
from zipfile import ZipFile
from shutil import move, rmtree
from torch.nn import Embedding, Linear

from utils import *

# Do not assign GPUs
os.environ['CUDA_VISIBLE_DEVICES']=''

def main():
    parser = argparse.ArgumentParser(
        description='Add new languages to an existing mBART model.'
    )
    parser.add_argument(
        "--pretrained_mbart", action='store_true',
        help="Whether to use a pre-trained mBART or the custom mBART built " \
            "from the specification of the arguments file."
    )
    sys_args, args_file = parser.parse_known_args()
    
    # Load configuration arguments
    try:
        args_file = args_file[0]
        with open(args_file) as f:
            args = json.loads(f.read(), object_hook=jsonParser)
    except:
        raise FileNotFoundError(
            'Provide a JSON with arguments as in "python prepare_data.py ' \
            'args.json"'
        )

    random.seed(args.seed)
    
    # Download the ParaCrawl 7.1 datasets specified in the argument
    # `args.paracrawl_languages`
    if args.download_paracrawl_datasets:
        for lang in args.paracrawl_languages:
            src_lang, tgt_lang = lang.split('-')
            corpora_path = args.root_dir + args.corpora_dir + lang
            source_filename = 'ParaCrawl.{}.{}'.format(
                lang, src_lang
            )
            target_filename = 'ParaCrawl.{}.{}'.format(
                lang, tgt_lang
            )
            
            # Check if file has been downloaded, otherwise download it
            if (
                not os.path.exists(corpora_path + 'temp/' + source_filename) or
                not os.path.exists(corpora_path + 'temp/' + target_filename)
            ):
                filename = corpora_path + '/temp/{}.txt.zip'.format(lang)
                print('Attempting to download the corpus for {}'.format(lang))
                # Create directory to store temporal files
                if not os.path.exists(corpora_path + '/temp/'):
                    os.makedirs(corpora_path + '/temp/')
                try:
                    download_path = 'https://opus.nlpl.eu/download.php?f=' \
                        'ParaCrawl/v7.1/moses/{}.txt.zip'.format(lang)
                    response = urllib.request.urlopen(download_path)
                    with open(filename, 'wb') as f:
                        while True:
                            tmp = response.read(1024)
                            if not tmp:
                                break 
                            f.write(tmp)
                    response.close()
                except urllib.error.HTTPError:
                    print('The download link {} seems to be invalid.'.format(
                        download_path
                    ))
                if os.path.exists(filename):
                    print('The file {} has been successfully downloaded'.format(
                        filename))
                else:
                    print('There was a problem downloading the file {}'.format(
                        filename
                    ))

                # Extract the corpus from the zip file
                with ZipFile(filename, 'r') as zipObj:    
                    zipObj.extractall(
                        path=corpora_path + '/temp/',
                        members=[source_filename, target_filename]
                    )

                # Delete the file after the extraction
                os.remove(filename)

            # Create a bilingual corpus
            source_file = open(corpora_path + '/temp/' + source_filename, 'r')
            target_file = open(corpora_path + '/temp/' + target_filename, 'r')
            with open(corpora_path + '/temp/corpus.all', 'w') as f:
                for src_line, tgt_line in zip(source_file, target_file):
                    line = '{}\t{}\n'.format(
                        src_line.rstrip(), tgt_line.rstrip()
                    )
                    f.write(line)
            source_file.close()
            target_file.close()
            
            # Shuffle the corpus
            lines = open(corpora_path + '/temp/corpus.all').readlines()
            random.shuffle(lines)
            open(corpora_path + '/temp/corpus.all','w').writelines(lines)

            # Create the inverse translation direction pair folder,
            # e.g. for es-en create en-es
            rev_lang = '{}-{}'.format(tgt_lang, src_lang)
            rev_corpora_path = args.root_dir + args.corpora_dir + rev_lang
            if not os.path.exists(rev_corpora_path):
                os.makedirs(rev_corpora_path)

            # Create dataset splits for train, development and test
            data_files = {
                'train': open(
                    corpora_path + '/train_{}.txt'.format(lang), 'w'
                ),
                'dev': open(
                    corpora_path + '/development_{}.txt'.format(lang), 'w'
                ),
                'test': open(
                    corpora_path + '/test_{}.txt'.format(lang), 'w'
                )
            }

            # The same as the previous step but for the opposite direction
            rev_data_files = {
                'train': open(
                    rev_corpora_path + '/train_{}.txt'.format(rev_lang), 'w'
                ),
                'dev': open(
                    rev_corpora_path + '/development_{}.txt'.format(rev_lang),
                    'w'
                ),
                'test': open(
                    rev_corpora_path + '/test_{}.txt'.format(rev_lang), 'w'
                )
            }

            # Actual splitting operation
            test_offset = args.development_size + args.test_size
            with open(corpora_path + '/temp/corpus.all', 'r') as f:
                i = 0
                for line in f:
                    src, tgt = line.split('\t')
                    rev_line = '{}\t{}\n'.format(tgt.rstrip(),src)
                    # First elements for development
                    if i < args.development_size:
                        data_files['dev'].write(line)
                        rev_data_files['dev'].write(rev_line)
                    # Next elements for test
                    elif i >= args.development_size and i < test_offset:
                        data_files['test'].write(line)
                        rev_data_files['test'].write(rev_line)
                    # Remaining elements for training
                    elif i >= test_offset:
                        data_files['train'].write(line)
                        rev_data_files['train'].write(rev_line)
                    i += 1

            for dic in [data_files, rev_data_files]:
                for v in dic.values():
                    v.close()

    """
    In this point, there should be a folder for each of the languages specified
    in the `args.languages` argument. Example with "en-es" and "nl-en".

    root/corpora/
        - en-es/
            -- train_en-es.txt
            -- development_en-es.txt
            -- test_en-es.txt
        - nl-en/
            -- train_nl-en.txt
            -- development_nl-en.txt
            -- test_nl-en.txt
    """

    # Check if each language has a data folder and if it has a train file
    # (necessary for the next steps)
    for lang in args.languages:
        lang_path = args.root_dir + args.corpora_dir + lang
        if not os.path.exists(lang_path):
            raise FileNotFoundError(
                'The language pair {} specified in the `args.languages` ' \
                'argument does not have a folder in the path {}.'.format(
                    lang, args.root_dir + args.corpora_dir
                )
            )
        if not os.path.exists(lang_path + '/train_{}.txt'.format(lang)):
            raise FileNotFoundError(
                'The language pair {} specified in the `args.languages` ' \
                'argument does not have a train file in the path {}.'.format(
                    lang, args.root_dir + args.corpora_dir + lang + '/'
                )
            )

    temp_path = args.all_corpus[:args.all_corpus.find('/')]
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    # Sample from each language to create a parallel (multilingual) corpus
    # to train a sentencepiece tokeniser
    with open(args.root_dir + args.all_corpus, 'w') as allcorpus:
        for lang in args.languages:
            src_lang, tgt_lang = lang.split('-')
            corpora_path = args.root_dir + args.corpora_dir + lang   
            
            # Sample `args.max_samples_per_lang` elements from each
            # language-pair. In case there are not enough samples, repeat them
            i = 0
            while i < args.max_samples_per_lang:
                train_file = open(corpora_path + '/train_{}.txt'.format(lang))
                for line in train_file:
                    allcorpus.write(line)
                    i += 1
                    
                    if i == args.max_samples_per_lang:
                        break
                train_file.close()

    # Check if any of the languages were not in mBART
    new_language_codes = []
    new_langs = {}
    for lang_pair in args.languages:
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

    # Train a tokeniser
    # See https://github.com/google/sentencepiece/blob/master/doc/options.md
    # for all the available options. `pad_id` must be set
    if os.path.exists(args.root_dir + args.tokeniser_dir):
        rmtree(args.root_dir + args.tokeniser_dir)
    os.makedirs(args.root_dir + args.tokeniser_dir)

    spm.SentencePieceTrainer.train(
        input=args.root_dir + args.all_corpus,
        model_prefix=args.root_dir + args.tokeniser_dir + args.tokeniser_prefix,
        vocab_size=args.vocabulary_size,
        character_coverage=args.tokeniser_character_coverage, 
        shuffle_input_sentence=True,
        input_sentence_size=args.tokeniser_input_sentence_size,
        #seed_sentencepiece_size=args.seed,
        unk_piece=args.unk_token, 
        bos_piece=args.bos_token, 
        eos_piece=args.eos_token, 
        pad_piece=args.pad_token,
        pad_id=3, # being 0, 1 and 2 the unk, bos and eos tokens, respectively
        control_symbols=','.join(new_language_codes)
    ) 

    if not os.path.exists(args.root_dir + args.pruned_model):
        os.makedirs(args.root_dir + args.pruned_model)

    # Save previous vocabulary (if any)
    vocab_path = args.root_dir + args.pruned_model + args.pruned_vocabulary_file
    if os.path.exists(vocab_path):
        move(
            vocab_path,
            args.root_dir + args.pruned_model + args.old_pruned_vocabulary_file
        )

    # Generate new vocabulary
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

    # Delete the file after the extraction
    os.remove(args.root_dir + args.all_corpus)

    # If the flag `--pretrained_mbart` is used, then an mBART model is loaded
    # otherwise a transformer with customisable architecture (specified in
    # the arguments file)
    if sys_args.pretrained_mbart:
        # Map the vocabulary of the original mBART and the new one, i.e.
        # words appearing in both have their word embedding copied to the
        # new model
        mapping, ft_dict, new_vocab = mapping_of_vocabularies(
            args,
            args.root_dir + args.pretrained_model_vocabulary_file,
            args.root_dir + args.pruned_model + args.pruned_vocabulary_file,
            new_language_codes = new_language_codes
        )
        model = MBartForConditionalGeneration.from_pretrained(
            args.pretrained_model
        )
        # Check if the specified embedding length is the same as mBART's
        mbart_embedding_length = model.get_input_embeddings().weight.shape[1]
        if mbart_embedding_length != args.embedding_length:
            raise ValueError(
                'The mBART embedding length {} does not match {} length ' \
                'specific by args.embedding_length. To use a custom length, ' \
                'disable the --pretrained_mbart option to prepare a custom ' \
                'transformer model.'.format(
                    mbart_embedding_length,
                    args.embedding_length
                )
            )
        # Copy the word embeddings to the new model using the mapping
        model = assign_embeddings_from_mapping(
            args,
            model,
            mapping,
            ft_dict,
            new_vocab
        )
    else:
        # Load a customised transformer
        model = custom_transformer(args, args.train_mode)

    # Save the model using the HuggingFace format and also the Pytorch standard
    torch.save(model, args.root_dir + args.pruned_model + args.model_filename)
    model.save_pretrained(args.root_dir + args.pruned_model)
    print('Model saved in the path {}'.format(
        args.root_dir + args.pruned_model
    ))
            
if __name__ == '__main__':
    main() 