import os

# Control what GPUs are being used
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import sys
import torch
import numpy as np
from time import time
from pytz import timezone
from tqdm.auto import tqdm
from datetime import datetime
from datasets.metric import Metric

from utils import *

# Used for time logging
fmt = '%Y-%m-%d %H:%M:%S %Z%z'

def main(args_file=None):
    # Load configuration arguments
    args = read_args(args_file if args_file is not None else sys.argv[1])

    if (
        args.num_train_epochs < 0 and 
        args.train_iterations < 0 and 
        args.train_updates < 0
    ):
        raise ValueError(
            'At least one of the following options must be >1: ' +
            'num_train_epochs, train_iterations, train_updates.'
        )
    
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

    save_args(args, args_file if args_file is not None else sys.argv[1])

    MODE = args.train_mode

    # Control randomness
    if args.seed is not None:
        set_seed(args.seed)

    # Initialises the training environment by loading the tokenizer, model and
    # the optimiser, among others
    (
        #tokenizer,
        data_collator,
        model,
        #optimiser,
        #loss_func,
        accelerator
    ) = initialise(args, MODE, 'pretrained', multigpu=args.multigpu)

    # Load metrics if the development set is used
    if args.development_iterations > 0 or args.development_epochs > 0:
        metrics = load_metrics(args)

    # Get the maximum amount of samples among all the corpora        
    max_samples = get_nb_samples(args)
    if max_samples <= 0:
        raise ValueError(
            'No sample was given for the finetuning.'
        )
    # Get the learning rate scheduler that modifies the LR during training
    #lr_scheduler = get_lr_scheduler(args, max_samples, optimiser)
    
    # If the epochs for training are set to -1, then force the training to 
    # go to infinity (other stop criterion needs to be used)
    num_train_epochs = args.num_train_epochs
    if args.num_train_epochs == -1:
        num_train_epochs = int(1e8)

    # Computes the total number of iterations per epoch
    iterations = max_samples // args.chunk_size
    if max_samples % args.chunk_size != 0:
        iterations += 1

    if accelerator.is_local_main_process:
        now = datetime.now(timezone(args.time_zone))
        print(now.strftime(fmt), '| Training start time')

    nb_iterations, train_updates = 0, 0

    # History arrays store the learning rate, loss and development BLEUs
    # They are later saved in separate files for their visualisation
    lr_history, loss_history = [], []
    bleu_history = {}
    for lang in args.languages:
        bleu_history[lang] = {}
        bleu_history[lang]['sacreBLEU'] = []

    # Training starts here
    for epoch in range(num_train_epochs):
        # Initialise timers for each epoch
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)
        epoch_start.record()
        # Print the current status
        if accelerator.is_local_main_process:
            per_device_batches = (
                max_samples // args.per_device_train_batch_size
            )
            if max_samples % args.per_device_train_batch_size != 0:
                per_device_batches += 1
            per_device_chunk_size = (
                args.chunk_size // torch.cuda.device_count()
            )
            now = datetime.now(timezone(args.time_zone))
            print(now.strftime(fmt), '| Epoch {} start, (per-device chunk ' \
                'size: {}, per-device batches: {}, total samples: {})'. format(
                    epoch+1,
                    per_device_chunk_size,
                    per_device_batches,
                    max_samples
                )
            )
        for iter in range(iterations):
            # Initialise timers for each iteration
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            # Set the model for training
            model.train()
            # Set the seeds in each epoch for reproducibility
            set_seed(args.seed)
            # Get data for the iteration
            dataloader, sampler = load_dataset(
                args, MODE, data_collator, model.tokeniser, accelerator, iter
            )
            # Set the epoch number for the sampler        
            if sampler is not None:
                sampler.set_epoch(epoch)
            # Variable to store the loss through the iteration (saved in the)
            # `loss_history` variable
            running_loss = 0.
            
            # Load the progress bar to be shown that also retrieves batches
            progress_bar = tqdm(
                enumerate(dataloader),
                disable=not accelerator.is_local_main_process
            )
            for step, batch in progress_bar:
                # Feedforward pass, computing the loss
                loss = model.feedforward_step(batch)
                
                running_loss += float(loss.detach())
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)

                # If the last batch is used or enough gradients have been
                # accumulated, update gradients
                if (
                    (step > 0 and
                    step % args.gradient_accumulation_steps == 0) or
                    step == len(dataloader) - 1 
                ):
                    model.optimiser_step()
                    train_updates += 1
                    
                    # After finishing all the training updates, stop
                    if train_updates == args.train_updates: 
                        break 
          
            # Iteration finished
            nb_iterations += 1
            loss_history.append(running_loss)
            lr_history.append(model.get_last_lr())

            # Compute metrics on the development set after 
            # `args.development_iterations` iterations
            if (
                args.development_epochs == -1 and
                args.development_iterations > 0 and
                nb_iterations % args.development_iterations == 0
            ):
                development_step(
                    args,
                    model,
                    model.tokeniser,
                    data_collator,
                    accelerator,
                    metrics,
                    bleu_history
                )

            # Save model and history arrays
            if accelerator.is_local_main_process:
                now = datetime.now(timezone(args.time_zone))
                print(now.strftime(fmt), '| Iteration {} finished'.format(
                    nb_iterations
                ))
                save_outputs(
                    args,
                    accelerator,
                    model,
                    nb_iterations,
                    lr_history,
                    loss_history,
                    bleu_history
                )
                model.save_model_state(epoch, iter)
            accelerator.wait_for_everyone()

            # Check if the training should stop
            if (
                nb_iterations == args.train_iterations or
                train_updates == args.train_updates
            ):
                end.record()
                if accelerator.is_local_main_process:
                    now = datetime.now(timezone(args.time_zone))
                    print(now.strftime(fmt), '| Stopped after {} iterations and {} updates in ' \
                        'epoch {}'.format(
                            nb_iterations, train_updates, epoch
                        )
                    )
                    print(now.strftime(fmt), '| Time to finish epoch {}: {:.2f}s'.format(
                        epoch+1, start.elapsed_time(end)/1000.0)
                    )
                sys.exit()
        # Compute metrics on the development set after 
        # `args.development_epochs` epochs
        if (
            args.development_iterations == -1 and
            args.development_epochs > 0 and
            (epoch+1) % args.development_epochs == 0
        ):
            development_step(
                args,
                model,
                model.tokeniser,
                data_collator,
                accelerator,
                metrics,
                bleu_history
            )
            
        # Save model and history arrays
        if accelerator.is_local_main_process:
            save_outputs(
                args,
                accelerator,
                model,
                nb_iterations,
                lr_history,
                loss_history,
                bleu_history,
                on_epoch_end=True
            )
            model.save_model_state(epoch, iter)
        epoch_end.record()
        if accelerator.is_local_main_process:
            now = datetime.now(timezone(args.time_zone))
            print(now.strftime(fmt), '| Time to finish epoch {}: {:.2f}s'.format(
                epoch+1, epoch_start.elapsed_time(epoch_end)/1000.0)
            )

def save_outputs(
    args,
    accelerator,
    model,
    nb_iterations: int,
    lr_history: List[float],
    loss_history: List[float],
    bleu_history,
    on_epoch_end: bool=False
) -> None:
    # Save model and history arrays

    # Save current model
    output_dir = args.root_dir + args.output_dir + \
        args.saved_models_dir + args.last_model_dir
    #model.save_model(accelerator, output_dir)

    # Create the directory to save plots if needed
    plots_path = args.root_dir + args.output_dir + args.plots_dir
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # Save the learning rate history
    np.save(
        plots_path + args.lr_history,
        np.asarray(lr_history)
    )

    # Save the loss history
    np.save(
        plots_path + args.loss_history,
        np.asarray(loss_history)
    )

    # Save the BLEU history
    for lang in args.languages:
        np.save(
            plots_path + '{}_{}'.format(
                lang, 'sacre' + args.bleu_history
            ),
            np.asarray(bleu_history[lang]['sacreBLEU'])
        )
    """   
    if not on_epoch_end:
        # Each N iterations, save the model in a folder that won't be
        # overwritten
        if (
            args.save_models_each_n_iters > 0 and
            nb_iterations % args.save_models_each_n_iters == 0
        ):
            output_dir = args.root_dir + args.output_dir + \
                args.saved_models_dir + '{}_iters/'.format(
                    nb_iterations 
                )
            model.save_model(accelerator, output_dir + 'model.pt') """

def development_step(
    args: Dict[str,Any], 
    model: MBartForConditionalGeneration, 
    tokenizer: MBartTokenizerFast, 
    data_collator: DataCollatorForSeq2Seq, 
    accelerator: Accelerator, 
    metrics: Dict[str,Metric],
    bleu_history: Dict[str,Dict[str,List[int]]]
) -> Dict[str,float]:
    """
    The function computes score for a development set per language-pair in a
    dictionary, containing as keys the languager-pair identifiers (e.g. 'en-es')
    and a float object representing the development metric defined by
    `args.development_metric`

    Parameters
    ----------    
    args: Dict[str,Any]
        Configuration dictionary (namedtuple) loaded from a JSON file
    model: HuggingFace MBartForConditionalGeneration Model
        Model loaded
    tokenizer: MBartTokenizerFast
        Tokenizer used to process the input text
    data_collator: DataCollatorForSeq2Seq
        Object that creates batches from lists of dataset elements
    accelerator: Accelerator
        Accelerator object
    metrics: Dict[str,Metric]
        Dictionary with pairs metric names and Metric objects
    bleu_history: Dict[str,Dict[str,List[int]]]
        Dictionary for each language-pair evaluated containing BLEU
        metrics as key and a list of values for each of them

    Returns
    ----------
    Dict[str,float]
        Dictionary with pairs of language-pairs and the resulting scores
        in the development set, e.g. "{'en-es': 0.1}"
    """
    model.eval()
    now = datetime.now(timezone(args.time_zone))
    if accelerator.is_local_main_process:
        print(now.strftime(fmt), '| Starting evaluation on the ' +
            'development set'
        )

    scores = {}
    metric = args.development_metric.lower()
    for lang in args.languages:
        # Control randomness
        set_seed(args.seed)

        # Load development data
        dev_dataloader, _ = load_dataset(
            args, args.development_mode, data_collator,
            tokenizer, accelerator, lang=lang
        )

        dev_progress_bar = tqdm(
            enumerate(dev_dataloader), 
            disable=not accelerator.is_local_main_process
        )
        #hyps, refs = [], []
        for _, dev_batch in dev_progress_bar: 
            with torch.no_grad():  
                generated_tokens = model.inference(dev_batch, accelerator)
                dev_batch = {
                    k: v.cuda(non_blocking=args.non_blocking) \
                        for k,v in dev_batch.items()
                }
                generated_tokens = (
                    accelerator.pad_across_processes(
                        generated_tokens,
                        dim=1,
                        pad_index=model.pad_token_id()
                    )
                )

                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need
                    #  to pad the labels too
                    labels = accelerator.pad_across_processes(
                        dev_batch['labels'], 
                        dim=1,
                        pad_index=model.pad_token_id()
                    )
                else:
                    labels = dev_batch['labels']

                generated_tokens = accelerator.gather(
                    generated_tokens
                ).cpu().numpy()
                labels = accelerator.gather(
                    labels
                ).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them
                    labels = np.where(
                        labels != -100, labels, tokenizer.pad_token_id
                    )

                # Decode from IDs to tokens and remove special tokens
                # such as BOS, EOS, UNK and PAD
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )
                metrics[metric].add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels
                )

        scores[lang] = metrics[metric].compute()['score'] # compute_metrics(refs, hyps) 
    now = datetime.now(timezone(args.time_zone))
    for lang in args.languages:
        if accelerator.is_local_main_process:
            print(now.strftime(fmt), '| {} BLEU score: {:.2f}'.format(
                lang, scores[lang]
            ))
        bleu_history[lang]['sacreBLEU'].append(
            scores[lang]
        )
    model.train()

if __name__ == "__main__":
    main()