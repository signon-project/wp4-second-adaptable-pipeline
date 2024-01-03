import os
import sys
import json
import torch
import shutil
import random
import numpy as np
import torch.distributed as dist
from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizerFast,
    MBartConfig,
    AdamW,
    get_scheduler,
    set_seed,
)
from shutil import move, copyfile
from datasets import load_metric
from datasets.metric import Metric
from accelerate import Accelerator
from collections import namedtuple
from fairseq.data import Dictionary
from torch.nn import Embedding, Linear
from torch.utils.data.sampler import Sampler
from torch.utils.data import RandomSampler, Dataset
from tokenizers import SentencePieceUnigramTokenizer
from typing import Any, Dict, Optional, Tuple, List, Union
from torch.utils.data.distributed import DistributedSampler
from transformers.data.data_collator import DataCollatorForSeq2Seq

from custom_dataloader import T, DataLoader
from dataset import MultiLingualDataset, new_language_code
from samplers import BucketBatchSampler, DistributedSamplerWrapper
from model import MultilingualModel, load_tokeniser

def set_seed(seed: int) -> None:
    """ 
    Sets a seed to avoid randomness.

    Parameters
    ----------    
    seed: int
        Integer to use as seed

    Returns
    ---------- 
    None
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def read_args(argv: str) -> namedtuple:
    """ 
    Loads the JSON file received as an argument in a namedtuple structure.

    Parameters
    ----------    
    argv: str
        Path to a configuration JSON file

    Returns
    ---------- 
    namedtuple
        A namedtuple object with the content of a JSON dictionary
    """
    try:
        with open(argv, 'r', encoding='utf-8') as f:
            args = json.loads(f.read(), object_hook=jsonParser)
    except:
        raise FileNotFoundError(
            'Provide a JSON with arguments as in "python {} args.json"'.format(
                os.path.basename(__file__)
            )
        )
    # Check if there is a mandatory argument missing
    check_args(args)
    return args

def jsonParser(dict : Dict[str,Any]) -> namedtuple:
    """ 
    Creates a namedtuple object from a JSON dictionary. Without a namedtuple,
    an attribute must be accessed as "args['batch_size']". Meanwhile, using
    namedtuple "args.batch_size" can be used.

    Parameters
    ----------    
    dict: Dict[str,Any]
        Configuration dictionary loaded from a JSON file

    Returns
    ---------- 
    namedtuple
        A namedtuple object with the content of a JSON dictionary
    """
    return namedtuple('X', dict.keys())(*dict.values())

class ConfigurationError(Exception):
    def __init__(self, missing_arg: str) -> None:
        super().__init__(f'The `args:{missing_arg}` argument does not exist.')

def check_args(args: namedtuple) -> None:
    FIELDS = [
    'seed', 'time_zone', 'root_dir', 'output_dir', 'corpora_dir', 'cache_dir',
    'saved_models_dir', 'last_model_dir',
    'predictions_dir', 'plots_dir', 'loss_history', 'lr_history',
    'bleu_history', 'file_offsets', 'model_filename', 'pretrained_model',
    'sentencepiece_model', 'pretrained_model_vocabulary_file',
    'original_pruned_vocabulary_file', 'all_corpus', 'pruned_vocabulary_file',
    'old_pruned_vocabulary_file', 'pruned_model', 'input_embeddings_file',
    'output_embeddings_file', 'old_input_embeddings_file',
    'old_output_embeddings_file', 'train_mode', 'development_mode',
    'evaluation_mode', 'download_paracrawl_datasets', 'vocabulary_size',
    'model_type', 'development_epochs', 'development_iterations', 'padding',
    'truncation', 'pad_to_max_length', 'dataloader_num_workers',
    'dataloader_pin_memory', 'dataloader_prefetch_factor',
    'ignore_pad_token_for_loss', 'chunk_size', 'test_size','development_size',
    'drop_last', 'batch_bucketing', 'bucket_size_multiplier', 'non_blocking',
    'always_save_metrics', 'metrics','development_metric', 'pruned_vocabulary',
    'use_fp16', 'use_cross_entropy_smoothing', 'cross_entropy_smoothing',
    'learning_rate', 'per_device_train_batch_size', 'num_train_epochs',
    'per_device_eval_batch_size', 'gradient_accumulation_steps',  
    'train_iterations', 'train_updates', 'adam_eps', 'adam_betas',
    'weight_decay', 'lr_scheduler_type', 'num_warmup_steps',
    'save_models_each_n_iters', 'resume_training', 'tokeniser_dir',
    'old_tokeniser_dir', 'tokeniser_prefix', 'tokeniser_character_coverage',
    'tokeniser_input_sentence_size', 'max_samples_per_lang', 'unk_token',
    'bos_token', 'eos_token', 'pad_token', 'encoder_layers', 'decoder_layers',
    'encoder_attention_heads',  'decoder_attention_heads', 'encoder_ffn_dim', 
    'decoder_ffn_dim', 'embedding_length', 'attention_dropout',
    'classifier_dropout', 'num_beams', 'train_max_length', 'val_max_length',
    'freeze_embeddings', 'paracrawl_languages', 'languages', 'language_codes'
    ]
    for field in FIELDS:
        if not field in args._fields:
            raise ConfigurationError(field)

def init_process(
    backend: Optional[str] = 'nccl'
) -> None:
    """ 
    Initialises the distributed environment. 

    Parameters
    ----------    
    backend: str
        Backend to use among 'nccl', 'gloo' and 'mpi'

    Returns
    ---------- 
    None
    """
    rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def initialise(
    args: namedtuple,
    mode: str,
    model_type: str = 'pretrained',
    multigpu: bool = False
):
    """ 
    Creates any necessary element for the training or evaluation of a model.
    It must be specified if the initialised environment is going to be used
    for training or evaluation with the `mode` argument, which takes the 
    `train` and `test` values.

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    mode: str
        String indicating if the environment should be initialised for training
        or for evaluation
    model_type: str
        Type of mBART model to load: pretrained, from the last checkpoint or
        from a specific iteration

    Returns
    ---------- 
    transformers.MBartTokenizerFast
        Tokenizer used to process the input text
    HuggingFace MBartForConditionalGeneration Model
        Model loaded
    [if mode=args.train_mode] torch Optimizer
        Optimiser used for the training
    Acccelerate Accelerator
        Accelerator object
    """
    if (
        mode != args.train_mode and
        mode != args.evaluation_mode and
        mode != args.development_mode
    ):
        raise ValueError('The `mode` argument in the `initialise` function ' \
            'must be initialised with the values of `args.train_mode`, ' \
            '`args.evaluation_mode` or `args.development_mode`.'
        ) 

    # The accelerator object is used to manage the use of GPUs and FP16
    accelerator = Accelerator(fp16=args.use_fp16)

    # Compute and save offsets for each train/validation/development sample
    # within the data files so that they can be quickly accessed
    if accelerator.is_local_main_process:
        compute_dataset_offsets(args, mode)
        # Compute development offsets when initialising the training environment
        if mode == args.train_mode:
            compute_dataset_offsets(args, args.development_mode)
    # Wait until the main process computes the offsets
    accelerator.wait_for_everyone()

    # Load the tokeniser, the mBART model and the data collator (used to manage
    # batches)
    nb_samples = get_nb_samples(args)
    if nb_samples <= 0:
        raise ValueError(
            'No sample was given for the finetuning.'
        )
    model = MultilingualModel(args, accelerator, mode, model_type, nb_samples, multigpu)

    label_pad_token_id = args.label_pad_token_id
    if not args.ignore_pad_token_for_loss:
        label_pad_token_id = model.pad_token_id
    # Initialise the data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=model.tokeniser,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    if mode == args.train_mode:
        return (
            data_collator,
            model,
            accelerator
        )
    elif mode == args.evaluation_mode:
        return (
            data_collator,
            model,
            accelerator
        )

def custom_transformer(
    args: namedtuple,
    mode: str
) -> MBartForConditionalGeneration:
    """ 
    Loads a custom transformer with the configuration obtained from the
    args.json file    

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    mode: str
        String indicating if the environment should be initialised for training
        or for evaluation
    Returns
    ---------- 
    MBartForConditionalGeneration
        mBART model loaded
    """
    config = MBartConfig(
        vocab_size=args.vocabulary_size,
        d_model=args.embedding_length,
        encoder_layers=args.encoder_layers,
        encoder_attention_heads=args.encoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_dim,
        decoder_layers=args.decoder_layers,
        decoder_attention_heads=args.decoder_attention_heads,
        decoder_ffn_dim=args.decoder_ffn_dim,
        classifier_dropout=args.classifier_dropout,
        attention_dropout=args.attention_dropout
    )
    model = MBartForConditionalGeneration(config)
    if args.train_max_length <= 0 or args.val_max_length <= 0:
        raise ValueError(
            'The `args.train_max_length` and `args.val_max_length` '
            'arguments must be positive and greater than 0.'
        )
    if mode == args.train_mode:
        model.config.max_length = args.train_max_length
    elif mode == args.evaluation_mode:
        model.config.max_length = args.val_max_length

    tokeniser = load_tokeniser(args, mode)
    model.resize_token_embeddings(len(tokeniser))
    return model

def load_dataset(
    args: namedtuple,
    mode: str,
    data_collator: DataCollatorForSeq2Seq,
    tokenizer: MBartTokenizerFast,
    accelerator: Accelerator,
    iter: int = 0,
    lang: str = ''
) -> Tuple[Dict[str,DataLoader], Dict[str,Sampler]]:
    """ 
    Creates a generator to load training or evaluation data on-the-fly
    depending on the `mode` argument, i.e. the DataLoader. Samplers to load
    data are also created and returned alongside DataLoaders in dictionaries,
    with the language pairs as keys. The necessary objects are passed through
    the Accelerator. iter controls which part of the dataset is loaded, i.e.
    the dataset is sliced using the `chunk_size` attribute contained within the
    args class and the `iter` slice is loaded.

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    mode: str
        String indicating if the environment should be initialised for training
        or for evaluation
    data_collator : DataCollatorForSeq2Seq
        Object that creates batches from lists of dataset elements
    tokenizer: MBartTokenizerFast
        Tokenizer used to process the input text
    accelerator: Accelerator
        Accelerator object
    iter: int
        Integer used to divide the dataset into iterations and select which
        slice to use depending on the iteration number
    lang: str
        Language pair to load (for the evaluation mode)

    Returns
    ---------- 
    Tuple[DataLoader, Sampler]
        A tuple comprising a DataLoader and a Sampler
    """

    if (
        mode != args.train_mode and
        mode != args.evaluation_mode and
        mode != args.development_mode
    ):
        raise ValueError(
            'The `mode` argument in the `initialise` function must be ' +
            'initialised as "train" or "test".'
        ) 
    
    # Load the data generator for train or validation
    dataset = load_onthefly_dataset(args, tokenizer, mode, iter, lang)
    
    # The dataloader is the final manager of batches and it needs to be called
    # in the training or evaluation loop
    dataloader, sampler = load_dataloader(
        args, mode, dataset, data_collator
    ) 
    # Prepare the dataloader and sampler using the Accelerator
    dataloader, sampler = accelerator.prepare(dataloader, sampler)
    return dataloader, sampler

def compute_dataset_offsets(args: namedtuple, mode: str) -> None:
    """ 
    In order to efficiently read specific lines from the datasets' files
    offsets for each sentence are computed (in bytes). That is, the i^th 
    sentence is associated to the i^th offset (from a specific dataset file)
    to directly jump there. This function computes and stores them.

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    mode: str
        String indicating if the environment should be initialised for training
        or for evaluation
    
    Returns
    ---------- 
    None
    """ 
    for lang in args.languages:
        src_lang, tgt_lang = lang.split('-')
        offsets_file = args.root_dir + args.corpora_dir + \
            src_lang + '-' + tgt_lang + '/' + mode + '_' + \
            src_lang + '-' + tgt_lang + '_' + args.file_offsets
        if not os.path.exists(offsets_file):
            filename = args.root_dir + args.corpora_dir + lang + \
                    '/{}_{}.txt'.format(mode, lang)
            line_offset = []
            offset = 0
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line_offset.append(offset)
                        # Accumulate the number of bytes that the i^th line
                        # is composed of
                        offset += len(line.encode('utf-8'))
            except FileNotFoundError:
                print('The file {} does not exist'.format(filename))
                raise
            np.save(offsets_file, np.asarray(line_offset))

def get_nb_samples(args: namedtuple, mode: str='train') -> int:
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

def load_onthefly_dataset(
    args: namedtuple,
    tokenizer: MBartTokenizerFast,
    mode: str,
    iter: int,
    lang: str=''
) -> Dict[str,Dataset]:
    """ 
    Loads a torch Dataset class instance used to load data on the fly.
    That is, data is not stored at the beginning in the RAM, it is loaded
    at training time. 

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    tokenizer: MBartTokenizerFast
        Tokenizer used to process the input text
    mode: str
        String indicating if the environment should be initialised for training
        or for evaluation
    iter: int
        Integer used to divide the dataset into iterations and select which
        slice to use depending on the iteration number
    lang: str
        Language pair to load (for the evaluation mode)

    Returns
    ---------- 
    Dataset
        torch Dataset
    """
    if mode == args.train_mode:
        dataset_files = {}
        for lang in args.languages:
            dataset = args.root_dir + args.corpora_dir + lang + \
                '/{}_{}.txt'.format(mode, lang)
            if not os.path.exists(dataset):
                raise FileNotFoundError(
                    'The language file {} was not found. If this language ' \
                    'is not going to be used remove it from the `languages` ' \
                    'attribute in the arguments file.'.format(dataset)
                )
            dataset_files[lang] = dataset
        dataset = MultiLingualDataset(
            args=args,
            filenames=dataset_files,
            tokenizer=tokenizer,
            mode=mode,
            iter=iter
        )
    elif mode == args.evaluation_mode or mode == args.development_mode:
        dataset_file = args.root_dir + args.corpora_dir + lang + \
                '/{}_{}.txt'.format(mode, lang)
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(
                'The language file {} was not found. If this language ' \
                'is not going to be used remove it from the `languages` ' \
                'attribute in the arguments file.'.format(dataset_file)
            )
        dataset = MultiLingualDataset(
            args=args,
            filenames={lang: dataset_file},
            tokenizer=tokenizer,
            mode=mode,
            iter=0
        )
    return dataset

def load_dataloader(
    args: namedtuple,
    mode: str,
    dataset: Dict[str,Dataset],
    data_collator: DataCollatorForSeq2Seq
) -> Tuple[Dict[str,DataLoader],Dict[str,Sampler]]:
    """ 
    Loads a torch DataLoader class instance used to manipulate the data
    of the torch Dataset class instance, e.g. it creates batches and provides
    them in the training loop.

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    mode: str
        String indicating if the environment should be initialised for training
        or for evaluation
    dataset: Dataset
        torch Dataset
    data_collator: DataCollatorForSeq2Seq
        Function used to pad inputs and labels

    Returns
    ---------- 
    DataLoader
        DataLoader (used to get batches)
    """
    rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = torch.cuda.device_count()
    sampler = None

    if mode == args.train_mode:
        # Bucket batching allows the batches to have sequences of
        # similar length to optimise the usage of GPUs
        if args.batch_bucketing:
            random_sampler = RandomSampler(dataset)
            sampler = BucketBatchSampler(
                random_sampler,
                batch_size=args.per_device_train_batch_size, 
                drop_last=args.drop_last,
                bucket_size_multiplier=args.bucket_size_multiplier
            )
            """ sampler = DistributedSamplerWrapper(
                sampler_,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            ) """
            dataloader = DataLoader(
                dataset, 
                collate_fn=data_collator, 
                num_workers=args.dataloader_num_workers,
                batch_sampler=sampler,
                pin_memory=args.dataloader_pin_memory,
                prefetch_factor=args.dataloader_prefetch_factor
            )
        else:
            """ sampler = DistributedSampler(
                dataset,
                rank=rank,
                num_replicas=world_size,
                shuffle=True,
                seed=args.seed
            ) """
            dataloader = DataLoader(
                dataset, 
                collate_fn=data_collator, 
                batch_size=args.per_device_train_batch_size,
                num_workers=args.dataloader_num_workers,
                #sampler=sampler,
                pin_memory=args.dataloader_pin_memory,
                prefetch_factor=args.dataloader_prefetch_factor
            )
    elif mode == args.evaluation_mode or mode == args.development_mode:     
        """ sampler = DistributedSampler(
            dataset,
            rank=rank,
            num_replicas=world_size,
            shuffle=False,
            seed=args.seed
        ) """
        dataloader = DataLoader(
            dataset, 
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.dataloader_num_workers,
            sampler=sampler,
            pin_memory=args.dataloader_pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor
        )
    return dataloader, sampler

def load_optimiser(
    args: namedtuple,
    model: MBartForConditionalGeneration
) -> AdamW:
    """ 
    Loads an AdamW optimiser with weight decay (optional, adjusted by
    args.weight_decay)

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    model: HuggingFace MBartForConditionalGeneration Model
        Model loaded

    Returns
    ---------- 
    torch Optimizer
        Optimiser used for the training
    """
    # Split weights in two groups, one with weight decay and the other without
    # it.
    no_decay = ["bias", "LayerNorm.weight"]
    params = model.named_parameters()
    optimiser_grouped_parameters = [{
        "params": [
            p for n, p in params if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [ p for n, p in params if any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    }]
    return AdamW(
        optimiser_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_eps,
        betas=args.adam_betas
    )

def postprocess_text(
    preds: List[str],
    labels: List[str]
) -> Tuple[List[str],List[List[str]]]:
    """ 
    Cleans the text and prepares it to compute metrics.

    Parameters
    ----------    
    args: Dict[str,Any]
        Configuration dictionary (namedtuple) loaded from a JSON file

    Returns
    ---------- 
    Tuple[List[str],List[List[str]]]
        Tuple of list and list of lists containing the predictions and labels
        trimmed
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def load_metrics(args: namedtuple) -> Dict[str,Metric]:
    """ 
    Loads the metrics specified in `args.metrics` and returns a dictionary
    containing as keys the names of the metrics and as values the associated
    Metric objects

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file

    Returns
    ---------- 
    Dict[str,Metric]
        Dictionary of lang-pairs and Metric object pairs
    """
    metrics = {}
    for metric in args.metrics:
        metrics[metric.lower()] = load_metric(
                metric.lower(),
                cache_dir=args.root_dir + args.cache_dir
            )
    return metrics

def load_dict(langs: List[str], path: str) -> Dictionary:
    """ 
    Loads a vocabulary from the file specific by the argument `path` and
    also adds the languages in the `langs` argument to the vocabulary.

    Parameters
    ----------    
    langs: List[str]
        List of language codes to be added to the vocabulary
    path: str
        Path to the vocabulary to be loaded

    Returns
    ---------- 
    Dictionary
        Dictionary object loaded with a vocabulary
    """
    d = Dictionary()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError:
        print(
            'There was an error loading the file {} to create a ' \
            'dictionary'.format(path)
        )
        raise
    for line in lines:
        word, field = line.rsplit(" ", 1)
        if not word in d:
            d.add_symbol(word)
    for l in langs:
        d.add_symbol(f'[{l}]')
    return d

# Helper functions from fastai
def reduce_loss(loss: torch.Tensor, reduction: str='mean'):
    """ 
    Reduces the loss tensor by averaging or by addition depending on the
    `reduction`argument.

    Parameters
    ----------    
    loss: torch.Tensor
        Loss tensor to be reduced
    reduction: str
        Method to reduce the loss tensor

    Returns
    ---------- 
    torch.Tensor
        Loss after reducing it or not
    """
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
       
# Implementation from fastai
# https://github.com/fastai/fastai2/blob/master/fastai2/layers.py#L338
class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing: float=0.1, reduction: str='mean'):
        super().__init__()
        self.ε,self.reduction = smoothing,reduction
    
    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = torch.nn.functional.nll_loss(
            log_preds, target, reduction=self.reduction
        )
        # (1-ε)* H(q,p) + ε*H(u,p)
        return (1-self.ε)*nll + self.ε*(loss/c) 

def mapping_of_vocabularies(
    args: namedtuple,
    source_vocab: str,
    target_vocab: str,
    new_language_codes: List[str]=[]
) -> Union[List[int], Dictionary, Dict[str, int]]:
    """ 
    Map an old vocabulary (source) to the new vocabulary (target) using two
    dictionaries. New language codes may be provided to include them. The
    function returns (i) an index mapping of correspondences between the
    source and the target vocabularies, (ii) a Dictionary object for the
    target vocabulary and (iii) the new vocabulary (word->index mapping).

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    source_vocab: str
        Path to the source vocabulary
    target_vocab: str
        Path to the target vocabulary
    new_language_codes: List[str]
        Optional list of new language codes to add

    Returns
    ---------- 
    mapping
        Array used to map indices from the target vocabulary (array position)
        to the source vocabulary (value of the position) in case of 
        correspondences
    ft_dict
        Object of class Dictionary containing the target vocabulary
    new_vocab
        New vocabulary built, i.e. dictionary mapping words to new indices
    """
    if not os.path.exists(source_vocab):
        copyfile(target_vocab, source_vocab)
    # Map new vocabulary to the old one
    pre_dict = load_dict(
        args.language_codes, 
        source_vocab
    )
    ft_dict = load_dict(
        args.language_codes + new_language_codes,
        target_vocab
    )

    words, mapping = [], []
    new_vocab = {}
    total, not_found = 0, 0
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        pre_index = pre_dict.index(word)
        # If the token did not exist in the previous model (UNK), add a -1
        if pre_index == pre_dict.unk():
            mapping.append(-1)
            not_found += 1
        else:
            mapping.append(pre_index)
        new_vocab[word] = i
        words.append(word)
        total += 1

    try:
        print('Tokens that were not in the previous model / new tokens :' \
            '{}/{} ({:.2f}%)'.format(
            not_found, total, 100*(float(not_found)/float(total))
        ))
    except ZeroDivisionError:
        print('There are no words in the new dictionary.')
        raise

    # Save the mapping
    path = args.root_dir + args.pruned_model + 'mapping.txt'
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(len(words)):
                f.write('{} {}\n'.format(words[i], mapping[i]))
    except IOError:
        print(
            'There was an error writing in the file {} to save the' \
            'mapping from the source to the target vocabulary'.format(path) 
        )
        raise
    # Save new pruned vocabulary
    path = args.root_dir + args.pruned_model + args.pruned_vocabulary_file
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for k,v in new_vocab.items():
                f.write('{} {}\n'.format(k,v))
    except IOError:
        print(
            'There was an error writing in the file {} to save the' \
            'new vocabulary created'.format(path) 
        )
        raise
    return mapping, ft_dict, new_vocab

def assign_embeddings_from_mapping(
    args: namedtuple,
    model,
    mapping,
    ft_dict,
    new_vocab
) -> MBartForConditionalGeneration:
    """ 
    Modified the embedding matrix of a model with a new vocabulary. The words
    that still appear in the new vocabulary have their word embeddings copied
    using the mapping. The new model is returned.

    Parameters
    ----------    
    args: namedtuple
        Configuration dictionary (namedtuple) loaded from a JSON file
    model: MBartForConditionalGeneration
        Model loaded
    mapping: List[int]
        Array used to map indices from the target vocabulary (array position)
        to the source vocabulary (value of the position) in case of 
        correspondences
    ft_dict: Dictionary
        Object of class Dictionary containing the target vocabulary
    new_vocab: Dict[str,int]
        New vocabulary built, i.e. dictionary mapping words to new indices

    Returns
    ---------- 
    model: MBartForConditionalGeneration
        Model loaded whose embedding matrix have been modified
    """
    # Path to the new embedding matrices
    input_embedding_weights = args.root_dir + args.pruned_model + \
        args.input_embeddings_file
    output_embedding_weights = args.root_dir + args.pruned_model + \
        args.output_embeddings_file

    # Save previous embeddings
    if os.path.exists(input_embedding_weights):
        move(
            input_embedding_weights, 
            args.root_dir + args.pruned_model + args.old_input_embeddings_file
        )
    if os.path.exists(output_embedding_weights):
        move(
            output_embedding_weights, 
            args.root_dir + args.pruned_model + args.old_output_embeddings_file
        )

    # `pre_tensor` contains the old embeddings
    pre_tensor: torch.Tensor = model.get_input_embeddings().weight

    # `ft_tensor` contains the new embeddings (to be filled)
    ft_tensor = torch.zeros(
        [
            len(ft_dict), args.embedding_length],
            dtype=pre_tensor.dtype,
            layout=pre_tensor.layout,
            device=pre_tensor.device,
    )
    for ft_i, pre_i in enumerate(mapping):
        # If the token did not exist in the original mBART model,
        # initialise a random word embedding. If the option
        # `random_new_embeddings` is True, initialise them to random
        if pre_i == -1:
            ft_tensor[ft_i] = torch.rand(1, args.embedding_length)
        # Otherwise, copy the original word embedding
        else:
            ft_tensor[ft_i] = pre_tensor[pre_i]
    torch.save(ft_tensor, input_embedding_weights)

    new_embeddings = Embedding(len(ft_dict), args.embedding_length)
    new_embeddings.weight.data = ft_tensor
    new_embeddings.padding_idx = new_vocab[args.pad_token]
    model.set_input_embeddings(new_embeddings)
    print(
        'New input embedding shape',
        model.get_input_embeddings().weight.shape
    )

    # Compute the new output weight matrix if it does not exist
    if os.path.exists(output_embedding_weights):
        ft_tensor = torch.load(output_embedding_weights)
    else:
        pre_tensor: torch.Tensor = model.get_output_embeddings().weight
        ft_tensor = torch.zeros(
            [
                len(ft_dict), args.embedding_length],
                dtype=pre_tensor.dtype,
                layout=pre_tensor.layout,
                device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            # If the token did not exist in the original mBART model,
            # initialise a random word embedding. If the option
            # `random_new_embeddings` is True, initialise them to random
            if pre_i == -1:
                ft_tensor[ft_i] = torch.rand(1, args.embedding_length)
            # Otherwise, copy the original word embedding
            else:
                ft_tensor[ft_i] = pre_tensor[pre_i]
        torch.save(ft_tensor, output_embedding_weights)
    m, n = ft_tensor.shape
    with torch.no_grad():
        # Instantiate new layer and copy the weights
        model.lm_head = Linear(n, m, bias=False)
        model.lm_head.weight.copy_(ft_tensor)
    print(
        'New output embedding shape',
        model.get_output_embeddings().weight.shape
    )

    # Modify the configuration of the model
    model.config.attention_dropout = args.attention_dropout
    model.config.classifier_dropout = args.classifier_dropout
    if args.train_max_length <= 0:
        raise ValueError(
            'The `args.train_max_length` argument must be positive and ' \
            'greater than 0.'
        )
    model.config.max_length = args.train_max_length
    tokeniser = load_tokeniser(args, args.train_mode)
    model.resize_token_embeddings(len(tokeniser))
    
    return model

def save_model_state(args, model, optimiser, lr_scheduler, epoch, iter):
    state = {
        'epoch': epoch,
        'iter': iter,
        'state_dict': model.state_dict(),
        'optimiser': optimiser.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }

    output_dir = args.root_dir + args.saved_models_dir + args.last_model_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(
        state, 
        output_dir + args.model_state_file
    )

def save_args(args, args_path):
    output_dir = args.root_dir + args.output_dir + \
        args.saved_models_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy2(args_path, output_dir + 'args.json')