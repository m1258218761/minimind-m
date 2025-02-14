import os
import torch
import warnings

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
from model.LMConfig import LMConfig
from model.model import Transformer

warnings.filterwarnings('ignore')


def init_model():
    device = 'cuda:0'
    # Do model patching and add fast LoRA weights
    # model_name_or_path = "./out/sft"
    tokenizer_name_or_path = "./dataset/tokenizer/my_tokenizer"
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

    lm_config = LMConfig()
    model = Transformer(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = init_model()
    training_config = DPOConfig(
        output_dir="./minimind_dpo",
        per_device_train_batch_size=8,
        remove_unused_columns=False,
        report_to="none",
        save_steps=2000,
        learning_rate=1e-5,
        num_train_epochs=1
    )

    dataset_path = './dataset/dpo/train_data.json'
    train_dataset = load_dataset('json', data_files=dataset_path)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_config,
        beta=0.1,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=512
    )
    dpo_trainer.train()
