# Adapted from: https://github.com/pytorch/fairseq/issues/2120#issuecomment-647429120
# See also: https://discuss.huggingface.co/t/pruning-a-model-embedding-matrix-for-memory-efficiency/5502/5

import os
import sys
import json
import torch
import argparse
from typing import List, Any, Dict
from torch.nn import Embedding, Linear
from torch.nn.parameter import Parameter
from transformers import MBartForConditionalGeneration

from utils import jsonParser, load_tokeniser, load_dict

os.environ['CUDA_VISIBLE_DEVICES']='' 

def main():
    # Load configuration arguments
    args = read_args(sys.argv[1])

    if not os.path.exists(
        args.root_dir + args.pretrained_model_vocabulary_file
    ):
        raise FileNotFoundError(
            'The file "{}" was not found, call `prepare_data.py` to download ' \
            'the file.'
        )
    if not os.path.exists(args.root_dir + args.original_pruned_vocabulary_file):
        raise FileNotFoundError(
            'The file "{}" was not found, call `prepare_data.py` to train ' \
            'the tokeniser and obtain the vocabulary first.'
        )

    base_path = args.root_dir + args.pruned_model
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Path to save the embeddings
    input_embedding_weights = args.root_dir + args.pruned_model + \
        args.input_embeddings_file
    output_embedding_weights = args.root_dir + args.pruned_model + \
        args.output_embeddings_file

    # Load the pre-trained mBART Model, the original dictionary and the
    # pruned one (ft_dict)
    langs = args.language_codes
    pre_dict = load_dict(
        langs, args.root_dir + args.pretrained_model_vocabulary_file
    )
    
    ft_dict = load_dict(
        langs, args.root_dir + args.pruned_model + args.pruned_vocabulary_file # args.original_pruned_vocabulary_file
    )
    model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model)
    
    # Map old vocabulary to the new one
    new_vocab = {}
    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        pre_index = pre_dict.index(word)
        # If the token did not exist in the original mBART model (UNK), add a -1
        if pre_index == pre_dict.unk():
            mapping.append(-1)
        else:
            mapping.append(pre_index)
        new_vocab[word] = i
    
    # Save new pruned vocabulary
    with open(
        args.root_dir + args.pruned_model + args.pruned_vocabulary_file,
        'w'
    ) as f:
        for k,v in new_vocab.items():
            f.write('{} {}\n'.format(k,v))
    
    # Compute the new embedding weight matrix if it does not exist
    if os.path.exists(input_embedding_weights):
        ft_tensor = torch.load(input_embedding_weights)
    else:
        pre_tensor: torch.Tensor = model.get_input_embeddings().weight
        ft_tensor = torch.zeros(
            [
                len(ft_dict), args.embedding_length],
                dtype=pre_tensor.dtype,
                layout=pre_tensor.layout,
                device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            # If the token did not exist in the original mBART model,
            # initialise a random word embedding
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
            # initialise a random word embedding
            if pre_i == -1:
                ft_tensor[ft_i] = torch.rand(1, args.embedding_length)
            # Otherwise, copy the original word embedding
            else:
                ft_tensor[ft_i] = pre_tensor[pre_i]
        torch.save(ft_tensor, output_embedding_weights)
    m,n = ft_tensor.shape
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
    model.config.max_length = args.train_max_length
    tokeniser = load_tokeniser(args, mode='train')
    model.resize_token_embeddings(len(tokeniser))
    
    # Save the model using the HuggingFace format and also the Pytorch standard
    torch.save(model, args.root_dir + args.pruned_model + 'model.pt')
    model.save_pretrained(args.root_dir + args.pruned_model)

if __name__ == "__main__":
    main()