import os
# Control what GPUs are being used
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import sys  
import json
import torch
import numpy as np
from tqdm.auto import tqdm

from utils import *

def main(args_file=None):
    # Load configuration arguments
    args = read_args(args_file if args_file is not None else sys.argv[1])

    if args.multigpu:
        # Get local rank of the process
        rank = int(os.environ.get('LOCAL_RANK', -1))
        world_size = torch.cuda.device_count()

        # Initialise the process
        init_process()
        torch.cuda.set_device(rank)
        print('Rank {}/{} process initialised.'.format(
            rank+1, world_size
        ))  

    MODE = args.evaluation_mode

    # Control randomness
    try:
        if args.seed is not None:
            set_seed(args.seed)
    except AttributeError:
        print('The `args:seed` argument does not exist.')
        raise

    # Control randomness
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    np.random.seed(args.seed)      
    torch.cuda.manual_seed_all(args.seed) 

    for lang in args.languages:
        
        # Initialises the evaluation environment by loading the tokenizer and
        # the model used, among others
        (
            #tokenizer,
            data_collator,
            model,
            accelerator
        ) = initialise(args, MODE, args.model_type)

        # Control randomness
        set_seed(args.seed)
        # Load the data
        dataloader, _ = load_dataset(
            args, MODE, data_collator, model.tokeniser, accelerator, lang=lang
        )

        progress_bar = tqdm(
            enumerate(dataloader), 
            disable=not accelerator.is_local_main_process
        )
        # Change the model to evaluation
        model.eval()
        # Clean the file where predictions are stored
        base_path = args.root_dir + args.output_dir + \
                    args.predictions_dir + lang + '/'
        if accelerator.is_local_main_process:
            if not os.path.exists(base_path):
                os.makedirs(base_path)
        accelerator.wait_for_everyone()
        # Clean the file where the predictions are written
        if accelerator.is_local_main_process:
            open(
                base_path + 'generated_text_{}_{}.txt'.format(
                    args.model_type, lang
                ), 
                'w',
                encoding='utf-8'
            ).close()
        for _, batch in progress_bar: 
            with torch.no_grad():  
                batch = {k: v.cuda() for k,v in batch.items()}
            
                # Get the token for the target language
                # used as the first token for the inputs of the decoder
                decoder_start_token_id = int(
                    batch['decoder_start_token_id'][0]
                )
                # Feedforward
                generated_tokens = accelerator.unwrap_model(model.model).generate(
                    batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    num_beams=args.num_beams,
                    decoder_start_token_id=decoder_start_token_id
                ) 
                
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens,
                    dim=1,
                    pad_index=model.tokeniser.pad_token_id
                )
                
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the
                    # labels too
                    labels = accelerator.pad_across_processes(
                        batch['labels'],
                        dim=1,
                        pad_index=model.tokeniser.pad_token_id
                    )
                else:
                    labels = batch['labels']

                generated_tokens = accelerator.gather(
                    generated_tokens
                ).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them
                    labels = np.where(
                        labels != -100, labels, model.tokeniser.pad_token_id
                    )

                # Decode from IDs to tokens and remove special tokens
                # such as BOS, EOS, UNK and PAD
                decoded_preds = model.tokeniser.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = model.tokeniser.batch_decode(
                    labels, skip_special_tokens=True
                )
                
                if accelerator.is_local_main_process:
                    # Save the generated text
                    with open(
                        base_path + 'generated_text_{}_{}.txt'.format(
                            args.model_type, lang
                        ),
                        'a',
                        encoding='utf-8'
                    ) as f:
                        for i in range(len(decoded_preds)):
                            if decoded_preds[i] == '':
                                print(
                                    'The predictions are empty. The model ' \
                                    'may have not learnt well.'
                                )
                                continue
                            if decoded_labels[i] == '':
                                print('There are no labels for a sample.')
                                raise ValueError
                            f.write('{}\t{}\n'.format(
                                decoded_preds[i], decoded_labels[i]
                            ))
                accelerator.wait_for_everyone()

if __name__ == '__main__':
    main() 