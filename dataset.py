import os
import torch
import numpy as np
from typing import Dict, List, Any
from collections import namedtuple
from torch.utils.data import Dataset
from transformers import MBartTokenizerFast
from transformers.models.mbart.modeling_mbart import shift_tokens_right 

def new_language_code(lang: str) -> str:
    """ 
    Returns the language code associated to a new language (not previously
    in mBART)

    Parameters
    ----------    
    lang: str
        Language that requires a language code

    Returns
    ---------- 
    str
        Language code

    Example
    ---------- 
    lang_code = new_language_code('ga')
    print(lang_code)
    >>> 'ga_GA'
    """
    return '{}_{}'.format(lang.lower(), lang.upper())

def get_max_samples(args: namedtuple, mode: str='train') -> int:
    """ 
    Get the number of samples from the largest corpus. As there may be corpus
    with fewer samples, the maximum is set first, then those datasets are 
    oversampled at training time. The function returns the maximum times the
    number of corpora.

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    mode: str
        String indicating if the environment should be initialised for training
        or for evaluation

    Returns
    ---------- 
    int
        Largest corpus' number of samples
    """ 
    sizes = []
    for lang in args.languages:
        src_lang, tgt_lang = lang.split('-')
        filename = args.root_dir + args.corpora_dir + \
                src_lang + '-' + tgt_lang + '/' + mode + '_' + \
                src_lang + '-' + tgt_lang + '_' + args.file_offsets
        try:
            sizes.append(len(np.load(filename)))
        except FileNotFoundError:
            print('The file {} does not exist'.format(filename))
            raise
    return max(sizes)*len(sizes)

class MultiLingualDataset(Dataset):
    """
    `MultiLingualDataset` inherits from torch.data.Dataset to create a class
    that loads and returns data points. The __init__ function loads the offsets
    required to obtain any random data point `i` as fast as possible in the
    __getitem__ function. It also computes the number of samples for the
    __len__ function.
    """
    def __init__(
        self,
        args: namedtuple,
        filenames: Dict[str, str],
        tokenizer: MBartTokenizerFast,
        mode: str,
        iter: int
    ) -> None:     
        """
        Arguments
        ----------    
        args: Dict[str,Any]
            Configuration dictionary (namedtuple) loaded from a JSON file
        filenames: Dict[str]
            Dictionary of files containing the input sentences
        MBartTokenizerFast
            Tokenizer used to process the input text
        mode: str
            String indicating if the train or evaluation mode is used
        iter: int
            Integer used to divide the dataset into iterations and select which
            slice to use depending on the iteration number
        
        Returns
        ---------- 
        None
        """
        if filenames is None or len(filenames.items()) == 0: 
            raise ValueError(
                'The `MultiLingualDataset` expects at least valid file name'
            )
        if tokenizer is None:
            raise ValueError(
                'The `MultiLingualDataset` expects a valid tokeniser, got ' +
                'None'
            )
        if iter < 0:
            raise ValueError(
                'The `iter` argument passed to the `MultiLingualDataset` ' +
                'class must be non-negative'
            )
        super(MultiLingualDataset).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        if mode == args.train_mode:
            self.max_length = args.train_max_length
        elif mode == args.evaluation_mode or mode == args.development_mode:
            self.max_length = args.val_max_length

        offsets_files = {}
        num_samples = 0

        # Number of samples per language pair
        nb_language_pairs = len(filenames.items())
        dataset_slice = args.chunk_size // nb_language_pairs

        # Check if there is enough data to get a whole chunk, otherwise
        # modify `dataset_slice`
        max_samples = get_max_samples(args, mode=mode)
        if dataset_slice*(iter+1) > (max_samples // nb_language_pairs):
            dataset_slice = max_samples//nb_language_pairs - dataset_slice*iter
                    
        if mode == args.train_mode:
            sizes = []
            for lang, _ in filenames.items():
                offsets_files[lang] = args.root_dir + args.corpora_dir + \
                    lang + '/' + mode + '_' + \
                    lang + '_' + args.file_offsets
                
                if not os.path.exists(offsets_files[lang]):
                    raise FileNotFoundError('The file {} does not exist.'.format(
                        offsets_files[lang]
                    ))
                sizes.append(len(np.load(offsets_files[lang])))
            if all(np.asarray(sizes) < dataset_slice):
                dataset_slice = max(sizes)
        
        self.list_of_files, self.dataset_offsets = [], []
        self.lang_codes, self.line_offsets = [], []
        for lang, filename in filenames.items():
            # Add the i^ith file name to the list to visit them in this order 
            # later
            self.list_of_files.append(filename)
            # Get the source and target languages and their codes (for mBART)
            src_lang, tgt_lang = lang.split('-')
            src_lang_code, tgt_lang_code = '', ''
            for lang_code in args.language_codes:
                if src_lang in lang_code:
                    src_lang_code = lang_code
                if tgt_lang in lang_code:
                    tgt_lang_code = lang_code 
            if src_lang_code == '':
                src_lang_code = new_language_code(src_lang)
            if tgt_lang_code == '':
                tgt_lang_code = new_language_code(tgt_lang)
            self.lang_codes.append([src_lang_code, tgt_lang_code])

            # For the evaluation, save the accumulated number of samples so
            # far (offset) to be used later to retrieve samples given an index
            if mode == args.evaluation_mode or mode == args.development_mode:
                self.dataset_offsets.append(num_samples)
            
            # File to save the file-offsets (to jump to a given line directly)
            offsets_files[lang] = args.root_dir + args.corpora_dir + \
                src_lang + '-' + tgt_lang + '/' + mode + '_' + \
                src_lang + '-' + tgt_lang + '_' + args.file_offsets
            
            if mode == args.train_mode:
                # Load only an slice of the dataset (from s to t)
                #s, t = args.chunk_size * iter, args.chunk_size * (iter+1)
                s, t = dataset_slice * iter, dataset_slice * (iter+1)
                arr = np.load(offsets_files[lang])
                max_iters = int(np.ceil(len(arr) / dataset_slice))
                if iter > 0 and iter >= max_iters:
                    iter_ = iter % max_iters
                    s, t = dataset_slice * iter_, dataset_slice * (iter_+1)
                # If it is beyond the size of the dataset, sample with
                # replacement enough elements to get (t-s) samples
                if len(arr) > s and len(arr) < t:
                    offsets = []
                    if s < len(arr):
                        offsets.extend(arr[s:])
                    l = np.random.choice(
                        arr, 
                        size=dataset_slice-len(offsets),
                        replace=True
                    )
                    # Concatenate with the samples that were left (if any)
                    if len(offsets) == 0:
                        self.line_offsets.append(l)
                    else:
                        self.line_offsets.append(np.concatenate(
                            (offsets, l)
                        ))
                else:
                    self.line_offsets.append(np.load(offsets_files[lang])[s:t])
                    num_samples += len(self.line_offsets[-1])
            elif mode == args.evaluation_mode or mode == args.development_mode:
                self.line_offsets.append(np.load(offsets_files[lang]))
                num_samples += len(self.line_offsets[-1])

        # Oversample data (each corpus should have the same size for the
        # sampling)
        if mode == args.train_mode:
            for i in range(len(self.line_offsets)):
                self.dataset_offsets.append(dataset_slice * i)
            self.length = dataset_slice * len(self.line_offsets)#args.chunk_size #max_size * len(self.line_offsets)
        elif mode == args.evaluation_mode or mode == args.development_mode:
            if num_samples == 0:
                raise ValueError('There is no sample to load')
            self.length = num_samples
        
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i : int) -> Dict[str,List[int]]:
        # Get the dataset index (from 0 to the number of language pairs)
        for j in range(len(self.dataset_offsets)-1, -1, -1):
            if i >= self.dataset_offsets[j]:
                k = j
                break
        # Within the k^th dataset, sample index j (subtract the accumulated
        # number of samples so far to start the count in 0)
        j = i - self.dataset_offsets[k]
        try:
            # Jump to the i^th line using the offset and read it
            with open(self.list_of_files[k], 'rb') as f:
                f.seek(self.line_offsets[k][j])
                line = f.readline().decode('utf-8') 
        except IndexError:
            raise
        # Split the line into the source and the target sentences
        srcsent, tgtsent = line.rstrip().split('\t')
        # Get the source and target languages' token IDs
        src_lang_code, tgt_lang_code = self.lang_codes[k]

        # Set the source language and tokenise the source sentence
        self.tokenizer.src_lang = src_lang_code
        item = self.tokenizer(
            srcsent,
            padding=self.args.padding, truncation=self.args.truncation,
            add_special_tokens=True, max_length=self.max_length 
        )
        # Set the target language and tokenise the target sentence
        self.tokenizer.tgt_lang = tgt_lang_code
        #tgt_land_code = self.tokenizer.vocab[tgt_lang_code]
        #self.tokenizer.set_tgt_lang_special_tokens([tgt_land_code])
        with self.tokenizer.as_target_tokenizer():
            label = self.tokenizer(
                tgtsent, 
                padding=self.args.padding, truncation=self.args.truncation,
                add_special_tokens=True, max_length=self.max_length
            )

        # The target language for the decoder for the evaluation
        if (
            self.mode == self.args.evaluation_mode or
            self.mode == self.args.development_mode
        ):
            item['decoder_start_token_id'] = torch.as_tensor(
                self.tokenizer.vocab[tgt_lang_code]
            )

        # Add the tokens for the target sentence in the dictionary created 
        # after tokenising the input
        item['labels'] = label['input_ids']
       
        return item 