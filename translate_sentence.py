import os
# Control what GPUs are being used
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='4'

import sys  
import torch
import numpy as np
from tqdm.auto import tqdm

from utils import * 

def translate_sentence(
    args,
    model,
    accelerator,
    tokeniser,
    txt, 
    src_lang,
    trg_lang
):
    decoder_start_token_id = tokeniser.get_vocab()[trg_lang]
    tokeniser.src_lang = src_lang
    item = tokeniser(txt, add_special_tokens=True)
    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model.model).generate(
            torch.as_tensor([item['input_ids']]).cuda(), 
            attention_mask=torch.as_tensor([item['attention_mask']]).cuda(), 
            num_beams=args.num_beams,
            decoder_start_token_id=decoder_start_token_id
        ) 
    translation = tokeniser.batch_decode(
        generated_tokens, skip_special_tokens=True
    )
    return translation

""" import sys
import torch
import numpy as np
from utils import initialise, read_args
args = read_args(sys.argv[1])
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True 
np.random.seed(args.seed)      
torch.cuda.manual_seed_all(args.seed) 
(
    _,
    model,
    accelerator
) = initialise(args, args.evaluation_mode, 'last', multigpu=False)
model.eval()
lang_mapping = {
    'spa': 'es_XX',
    'eng': 'en_XX',
    'gle': 'ga_GA',
    'nld': 'nl_XX',
    'dut': 'nl_XX'
}
txt = 'My name is Adrian'
src_lang = 'eng'
trg_lang = 'spa'
translation = translate_sentence(
    args,
    model,
    accelerator,
    model.tokeniser,
    txt,
    lang_mapping[src_lang.lower()],
    lang_mapping[trg_lang.lower()]
)
print(translation) """