import os
import torch

from transformers import (
    MBartForConditionalGeneration,
    MBartConfig,
    MBartTokenizerFast,
    get_scheduler,
    AdamW
)
from tokenizers import SentencePieceUnigramTokenizer
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from dataset import new_language_code

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

def load_tokeniser(args, mode='train'):
        """ 
        Loads an MBart tokenizer

        Parameters
        ----------    
        args: namedtuple
            Configuration dictionary (namedtuple) loaded from a JSON file
        mode: str
            String indicating if the environment should be initialised for training
            or for evaluation

        Returns
        ---------- 
        MBartTokenizerFast
            Tokenizer used to process the input text
        """
        # If the model is pruned, the tokeniser needs to be adapted
        if args.pruned_vocabulary:
            # Check if all the new languages are in mBART, otherwise add
            # new lang token
            new_langs = {}
            for lang_pair in args.languages:
                for lang in lang_pair.split('-'):
                    new_langs[lang] = True

            for lang in new_langs.keys():
                for lang_code in args.language_codes:
                    if lang in lang_code: 
                        new_langs[lang] = False
            
            # If any language is left, then add special language code token
            # to the model
            new_codes = []
            for lang in new_langs.keys():
                if new_langs[lang]:
                    new_code = new_language_code(lang)
                    new_codes.append(new_code)

            # Create custom tokeniser to load a pruned vocabulary
            _tokeniser = SentencePieceUnigramTokenizer.from_spm(
                args.root_dir + args.tokeniser_dir + args.tokeniser_prefix + \
                    '.model'
            )

            # Use the custom tokeniser to initialise an mBART tokeniser
            tokeniser = MBartTokenizerFast(
                vocab_file = args.root_dir + args.pruned_model + \
                    args.pruned_vocabulary_file,
                use_fast=True,
                **{'tokenizer_object': _tokeniser}
            )

            # Add language codes that are not in the default mBART
            if len(new_codes) > 0:
                special_tokens_dict = {'additional_special_tokens': new_codes}
                tokeniser.add_special_tokens(special_tokens_dict)
        
            # Include the special tokens in the tokenizer. If they do not exist,
            # a new entry is added.
            tokeniser.add_special_tokens({
                'unk_token': args.unk_token,
                'bos_token': args.bos_token,
                'eos_token': args.eos_token,
                'pad_token': args.pad_token
            })
            
            # Set the maximum number of tokens per sequence
            if args.train_max_length <= 0 or args.val_max_length <= 0:
                raise ValueError(
                    'The `args.train_max_length` and `args.val_max_length` '
                    'arguments must be positive and greater than 0.'
                )
            if mode == args.train_mode:
                tokeniser.model_max_length = args.train_max_length
            elif mode == args.evaluation_mode or mode == args.development_mode:
                tokeniser.model_max_length = args.val_max_length
        else:
            tokeniser = MBartTokenizerFast.from_pretrained(
                args.pretrained_model, use_fast=True
            )
        return tokeniser
       
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

class MultilingualModel:
    def __init__(self, args, accelerator, mode, model_type, nb_samples, multigpu=False):
        self.args = args
        self.mode = mode
        self.model_type = model_type
        self.nb_samples = nb_samples
        self.multigpu = multigpu
        self.device = accelerator.device

        # Load the tokeniser
        self._load_tokeniser()

        # Loads the model, creating the `self.model` variable
        self._load_model()

        # Load an optimiser and a custom loss function
        self.optimiser, self.loss_func = None, None
        if self.mode == args.train_mode:
            # Custom loss function, only used if `args:use_cross_entropy_smoothing`
            # is True
            self.loss_func = LabelSmoothingCrossEntropy(
                smoothing=self.args.cross_entropy_smoothing,
            )
            # Load the optimiser and pass it through the accelerator
            self._load_optimiser()  
            self.optimiser = accelerator.prepare(self.optimiser)

            # Load the learning rate scheduler
            self._load_lr_scheduler()

        self.model = accelerator.prepare(self.model)

    def tokeniser(self):
        return self.tokeniser

    def custom_loss_func(self):
        return self.loss_func

    def optimiser(self):
        return self.optimiser

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def pad_token_id(self):
        return self.tokeniser.pad_token_id

    def _load_model(self) -> None:
        """ 
        Loads a model depending on the `model_type` argument, being the available
        options:
            - `pretrained`: loads the original pre-trained model from Facebook,
                    with a pruned vocabulary if `args.pruned_vocabulary` is True
            - `lang`: loads a language-specific model
            - `x_iters`: loads the model from the iteration x

        Parameters
        ----------    
        self

        Returns
        ---------- 
        None
        """
        path = ''
        self.model = None
        
        # A pretrained model on monolingual corpora
        if self.model_type == 'pretrained' and not self.args.resume_training:
            # For this option, the model needs to be pruned first
            if self.args.pruned_vocabulary:
                path = self.args.root_dir + self.args.pruned_model
            else:
                self.model = MBartForConditionalGeneration.from_pretrained(
                    self.args.pretrained_model
                )
                self.model.config.attention_dropout = self.args.attention_dropout
                self.model.config.classifier_dropout = self.args.classifier_dropout
                if self.args.train_max_length <= 0 or self.args.val_max_length <= 0:
                    raise ValueError(
                        'The `args.train_max_length` and `args.val_max_length` '
                        'arguments must be positive and greater than 0.'
                    )
                if self.mode == self.args.train_mode:
                    self.model.config.max_length = self.args.train_max_length
                elif self.mode == self.args.evaluation_mode:
                    self.model.config.max_length = self.args.val_max_length
        # Load the last model saved  
        elif self.model_type == 'last' or self.args.resume_training:
            self._custom_transformer()
            path = self.args.root_dir + self.args.output_dir + \
                self.args.saved_models_dir + self.args.last_model_dir
            self.load_state_dict(os.path.join(path, self.args.model_state_file))
        # Load a model from a specific training iteration (if it exists)
        elif 'iters' in self.model_type:
            path = self.args.root_dir + self.args.output_dir + \
                self.args.saved_models_dir + self.args.model_type + '/'
            self.load_state_dict(os.path.join(path, self.args.model_state_file))
        else:
            self._custom_transformer()
            
        # If the model has not been previously loaded
        if self.model is None:
            try:
                self.model = torch.load(path + 'model.pt')
            except FileNotFoundError:
                print('Cannot load model {}, does it exist? '.format(
                    path + 'model.pt'
                ))
                raise

        # Resizes the model embedding matrix according to the tokeniser's
        # vocabulary
        self.model.resize_token_embeddings(len(self.tokeniser))
        self.model.config.pad_token_id = self.tokeniser.pad_token_id

        # Freeze the embeddings if necessary
        if self.mode == 'train':
            if self.args.freeze_embeddings:
                for param in self.model.get_input_embeddings().parameters():
                    param.requires_grad = False
                for param in self.model.get_output_embeddings().parameters():
                    param.requires_grad = False
        
        # For training in multi-GPU, wrap the model with the
        # DistributedDataParallel wrapper
        if self.multigpu:
            rank = int(os.environ.get('LOCAL_RANK', -1))
            self.model = self.model.to(rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank], output_device=rank
            )
        else:
            self.model = self.model.cuda()

    def _custom_transformer(self) -> None:
        """ 
        Loads a custom transformer with the configuration obtained from the
        args.json file    

        Parameters
        ----------    
        self
       
        Returns
        ---------- 
        None
        """
        config = MBartConfig(
            vocab_size=self.args.vocabulary_size,
            d_model=self.args.embedding_length,
            encoder_layers=self.args.encoder_layers,
            encoder_attention_heads=self.args.encoder_attention_heads,
            encoder_ffn_dim=self.args.encoder_ffn_dim,
            decoder_layers=self.args.decoder_layers,
            decoder_attention_heads=self.args.decoder_attention_heads,
            decoder_ffn_dim=self.args.decoder_ffn_dim,
            classifier_dropout=self.args.classifier_dropout,
            attention_dropout=self.args.attention_dropout
        )
        self.model = MBartForConditionalGeneration(config)
        if self.args.train_max_length <= 0 or self.args.val_max_length <= 0:
            raise ValueError(
                'The `args.train_max_length` and `args.val_max_length` '
                'arguments must be positive and greater than 0.'
            )
        if self.mode == self.args.train_mode:
            self.model.config.max_length = self.args.train_max_length
        elif self.mode == self.args.evaluation_mode:
            self.model.config.max_length = self.args.val_max_length

        self.model.resize_token_embeddings(len(self.tokeniser))

    def _load_lr_scheduler(self) -> None:
        """
        Loads and returns the learning rate scheduler depending on the value of
        `args.lr_scheduler_type` and the amount of steps/iterations/epochs.

        Parameters
        ----------    
        self

        Returns
        ---------- 
        None
        """
        world_size = torch.cuda.device_count()
        batch_size = min(
            self.args.per_device_train_batch_size, self.args.chunk_size
        )
        
        # Compute the number of updates per iteration
        train_steps = self.args.chunk_size // world_size
        if self.args.chunk_size % world_size != 0:
            train_steps += 1
        add = 0
        if train_steps % batch_size != 0:
            add = 1
        train_steps //= batch_size
        train_steps += add
        add = 0
        if train_steps % self.args.gradient_accumulation_steps != 0:
            add += 1
        train_steps //= self.args.gradient_accumulation_steps
        train_steps += add
    
        # Three possible scenarios: fixed iterations, epochs or updates. Decide
        # which of the three will stop the training earlier.
        elems = []
        if self.args.train_iterations != -1:
            max_train_steps_iters = train_steps * self.args.train_iterations
            elems.append(max_train_steps_iters)
        if self.args.num_train_epochs != -1:
            nb_iters_per_epoch = self.nb_samples // self.args.chunk_size
            if self.nb_samples % self.args.chunk_size != 0:
                nb_iters_per_epoch += 1
            nb_iters_per_epoch *= train_steps
        
            max_train_steps_epoch =  nb_iters_per_epoch * self.args.num_train_epochs
            elems.append(max_train_steps_epoch)
        if self.args.train_updates != -1:
            elems.append(self.args.train_updates)

        # The one finishing first sets the maximum number of training steps
        max_train_steps = int(min(elems)) 
        
        # To avoid division by zero
        num_warmup_steps = self.args.num_warmup_steps
        if max_train_steps == num_warmup_steps:
            num_warmup_steps -= 1
        
        # Configure the learning rate scheduler
        # Names: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 
        # 'constant', 'constant_with_warmup'
        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimiser,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps
        )

    def _load_tokeniser(self) -> None:
        self.tokeniser = load_tokeniser(self.args, self.mode)

    def _load_optimiser(self) -> None:
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
        params = self.model.named_parameters()
        optimiser_grouped_parameters = [{
            "params": [
                p for n, p in params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": self.args.weight_decay,
        },
        {
            "params": [ p for n, p in params if any(nd in n for nd in no_decay)],
            "weight_decay": self.args.weight_decay,
        }]
        self.optimiser = AdamW(
            optimiser_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_eps,
            betas=self.args.adam_betas
        )

    def feedforward_step(self, batch):
        # Send the input tensors to the CUDA device
        batch = {
            k: v.cuda(non_blocking=self.args.non_blocking) \
                for k,v in batch.items()
        }
        
        # Remove the `labels` field before the feedforward pass
        labels = batch.pop('labels')
        # Prepare the decoder inputs (for the teacher forcing)
        decoder_input_ids = shift_tokens_right(
            torch.as_tensor(labels),
            self.tokeniser.pad_token_id
        )
        
        # Feedforward pass
        outputs = self.model(
            **batch,
            labels=labels,
            decoder_input_ids=decoder_input_ids
        ) 

        if self.args.use_cross_entropy_smoothing:
            # Replace the -100 for the padding token ID
            if self.args.ignore_pad_token_for_loss:
                labels[labels==-100] = self.tokeniser.pad_token_id
            # Transform the labels to a one-hot encoding representation
            # i.e. a vocabulary size array with a 1 in the position of a
            # word that is contained in the i^th label
            N, _, M = outputs.logits.shape
            y_onehot = torch.LongTensor(N, M).to(self.device)
            y_onehot.zero_()
            y_onehot.scatter_(1, labels, 1)
            # Compute the loss
            loss = self.loss_func(outputs.logits, y_onehot) 
        else:
            loss = outputs.loss
        return loss

    def inference(self, batch, accelerator):
        batch = {
            k: v.cuda(non_blocking=self.args.non_blocking) \
                for k,v in batch.items()
        }
        # Get the token for the target language
        # used as the first token for the inputs of 
        # the decoder
        decod_start_token_id = int(
            batch['decoder_start_token_id'][0]
        )
        # Feedforward
        #mask = dev_batch['attention_mask'][i:i+1]
        generated_tokens = accelerator.unwrap_model(
            self.model).generate(
            batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            num_beams=self.args.num_beams,
            decoder_start_token_id=decod_start_token_id
        ) 
        return generated_tokens

    def optimiser_step(self):
        self.optimiser.step()
        self.lr_scheduler.step()
        self.optimiser.zero_grad()

    def save_model_state(self, epoch, iter):
        state = {
            'epoch': epoch,
            'iter': iter,
            'state_dict': self.model.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'scheduler': self.lr_scheduler.state_dict()
        }

        output_dir = self.args.root_dir + self.args.output_dir + self.args.saved_models_dir + self.args.last_model_dir 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(
            state, 
            output_dir + self.args.model_state_file
        )

    def load_state_dict(self, path):
        self.model.load_state_dict(torch.load(path)['state_dict'])

    def save_model(self, accelerator, path):
        if not os.path.exists(path):
            os.makedirs(path)
        unwrapped_model = accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            path, save_function=accelerator.save
        )
        torch.save(unwrapped_model, path + 'model.pt')

    def get_last_lr(self):
        return self.lr_scheduler.get_last_lr()