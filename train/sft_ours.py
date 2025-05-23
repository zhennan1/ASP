import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
# from trainer import LisaTrainer
import transformers
import trl
import wandb
import torch
from peft import LoraConfig, get_peft_model, PeftModel
wandb.init(mode="disabled")
print("THIS IS SFT TRAINING")

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    block_size: int = field(default=8192)
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    rho: float = field(default=0)
    opt_method: str = field(default="sft")
    previous_lora: str = field(default="")
    reasoning_step: str = field(default=2)
    alignment_step: str = field(default=50)
    previous_lora: str = field(default="")
    system_evaluate: str = field(default="False")
    mode: str = field(default="None")
    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity
import re
def convert_format(example):
    text += "<｜end▁of▁sentence｜>"
    example['text'] = text

    if "<|im_start|>user" in text:
        raise ValueError("User token found in text")
    if "<|im_start|>assistant" in text:
        raise ValueError("Assistant token found in text")
    if "<|im_start|>think" in text:
        raise ValueError("Think token found in text")
    if "<|im_start|>answer" in text:
        raise ValueError("Answer token found in text")
    if "<|im_end|>" in text:
        raise ValueError("End token found in text")
    return example

def convert_format_for_training(example, dataset_name, mode):
    if dataset_name == "UWNSL/SafeChain":
        if mode == "safetune":
            example['text'] = "<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<｜User｜>"+example['instruction']+"<｜Assistant｜><think>\nLet's think about safety first." #safetune
        elif mode == "refusetune":
            example['text'] = "<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<｜User｜>"+example['instruction']+"<｜Assistant｜><think>\nI can't answer this question for safety reason." #safetune
        elif mode == "safechain":
            example['text'] = "<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<｜User｜>"+example['instruction']+"<｜Assistant｜><think>"+example['response']+"<｜end▁of▁sentence｜>" #safechain
    elif dataset_name == "fwnlp/data-advisor-safety-alignment" and mode == "direct_refusal":
        example['text'] = "<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<｜User｜>"+example['prompt']+"<｜Assistant｜><think>I should not answer this question!</think>"+example['response']+"<｜end▁of▁sentence｜>"
    else: raise ValueError(f"Dataset name not recognized in qwen: {dataset_name}, mode: {mode}")

    return example

def convert_format_for_training_llama(example, dataset_name, mode):
    if dataset_name == "UWNSL/SafeChain":
        if mode == "safetune":
            example['text'] = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"+example['instruction']+"<｜Assistant｜><think>\nLet's think about safety first." #safetune
            print("safetune")
        elif mode == "refusetune":
            example['text'] = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"+example['instruction']+"<｜Assistant｜><think>\nI can't answer this question for safety reason." #safetune
            print("refusetune")
        elif mode == "harmfultune":
            example['text'] = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"+example['instruction']+"<｜Assistant｜><think>\nI think it's harmful request." #safetune
            print("harmfultune")
        elif mode == "safetyclosetune":
            example['text'] = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"+example['instruction']+"<｜Assistant｜><think>\nLet's think about safety first.\n</think>" #safetune
            print("safetyclosetune")
        elif mode == "closetune":
            example['text'] = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"+example['instruction']+"<｜Assistant｜><think>\n\n</think>" #safetune
            print("closetune")
        elif mode == "safechain":
            example['text'] = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"+example['instruction']+"<｜Assistant｜><think>"+example['response']+"<｜end▁of▁sentence｜>"#safechain
    elif dataset_name == "fwnlp/data-advisor-safety-alignment" and mode == "direct_refusal":
            example['text'] = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"+example['prompt']+"<｜Assistant｜><think>I should not answer this question!</think>"+example['response']+"<｜end▁of▁sentence｜>"
    else: raise ValueError(f"Dataset name not recognized in llama3: {dataset_name}, mode: {mode}")

    return example

def convert_format_for_training2(example, tokenizer):
    generation_tokens = tokenizer(example['generations'][0], add_special_tokens=False)["input_ids"]
    generation_text = tokenizer.decode(generation_tokens, skip_special_tokens=True)
    example['text'] = "<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<｜User｜>"+example['problem']+"<｜Assistant｜><think>"+generation_text
    return example

def convert_format_for_training2_llama(example, tokenizer):
    generation_tokens = tokenizer(example['generations'][0], add_special_tokens=False)["input_ids"]
    generation_text = tokenizer.decode(generation_tokens, skip_special_tokens=True)
    example['text'] = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"+example['problem']+"<｜Assistant｜><think>"+generation_text
    return example

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "cache_dir":"cache" , "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        kwargs = { "use_cache": False, "torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name,**kwargs)
     
    model = model.to(torch.bfloat16)
    model.eval()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    dataset2 = load_dataset("open-r1/OpenR1-Math-220k", "default")



    if "SafeChain" in config.train_file_path:
        dataset = load_dataset(config.train_file_path)
    elif "wildjailbreak" in config.train_file_path:
        dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
        dataset['train'] = dataset['train'].filter(lambda x: 'vanilla_harmful' == x['data_type'] )
        print("Length is determined now", len(dataset['train']))
    elif "safety-alignment" in config.train_file_path:
        dataset = load_dataset("fwnlp/data-advisor-safety-alignment")

    dataset['train'] = dataset['train'].filter(lambda x: 'harmful' in x['label'] )
    dataset2['train'] = dataset2['train'].filter(lambda x: x['correctness_math_verify'][0] )

    


    if "Llama" in config.model_name:
        print("llama")
        dataset['train'] = dataset['train'].select(range(40))
        dataset2['train'] = dataset2['train'].select(range(40))
        dataset = dataset.map(convert_format_for_training_llama, fn_kwargs={"dataset_name": config.train_file_path, "mode": config.mode})
        dataset2 = dataset2.map(convert_format_for_training2_llama, fn_kwargs={"tokenizer": tokenizer})
    elif 'Qwen' in config.model_name:
        print("qwen")
        dataset['train'] = dataset['train'].select(range(400))
        dataset2['train'] = dataset2['train'].select(range(0))
        dataset = dataset.map(convert_format_for_training, fn_kwargs={"dataset_name": config.train_file_path, "mode": config.mode})
        dataset2 = dataset2.map(convert_format_for_training2, fn_kwargs={"tokenizer": tokenizer})

    else: raise ValueError(f"Model name not recognized: {config.model_name}")

    instruction_template = "<｜User｜>"
    response_template = "<｜Assistant｜>"

    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    if "Qwen" in config.model_name:
        args.max_steps = 100
        print("Qwen total steps", 100)
    elif "Llama" in config.model_name:
        args.max_steps = 20
        print("Llama total steps", 20)
    
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    args.rho=config.rho

    dataset["train"] = dataset["train"].map(lambda x: {"text": x["text"]})
    dataset2["train"] = dataset2["train"].map(lambda x: {"text": x["text"]})


    train_dataset = concatenate_datasets([dataset["train"], dataset2["train"]]).shuffle(seed=42)
    eval_dataset = dataset["test"] if "test" in dataset else train_dataset


    if config.opt_method=="sft":
        trainer = trl.SFTTrainer(
            model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            data_collator=collator,
            formatting_func=lambda x: x["text"],
        )
    else:
        import sys
        sys.exit()
        
        
    num_train_samples = len(dataset['train'])
    num_train_epochs = args.num_train_epochs
    train_batch_size = args.per_device_train_batch_size*8
    gradient_accumulation_steps = args.gradient_accumulation_steps
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    total_steps = num_train_epochs * (num_train_samples // effective_batch_size)
    print(total_steps)
    from transformers import TrainerCallback
    class GPUTimeCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic = 0
            self.record_time = 0
        
        def on_step_begin(self, args, state, control, **kwargs):
            state.start_event = torch.cuda.Event(enable_timing=True)
            state.end_event = torch.cuda.Event(enable_timing=True)
            state.start_event.record()
    

        def on_step_end(self, args, state, control, **kwargs):
            state.end_event.record()
            torch.cuda.synchronize()
            step_time = state.start_event.elapsed_time(state.end_event)
            self.average_statistic =  (self.average_statistic* self.record_time +step_time) / (self.record_time+1)  
            self.record_time +=1
            if self.record_time%50==0:
                # print(f"Step {state.global_step}: {self.average_statistic*self.record_time / 1000:.2f} seconds (GPU time)")
                print("Estimated total time {} (h)".format(self.average_statistic*total_steps/ 1000/3600))
                
    
    class GPUMemoryCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic_memory = 0
            self.record_time_memory = 0
        
        def on_step_begin(self, args, state, control, **kwargs):
            state.start_memory = torch.cuda.memory_reserved()
            # print(self.record_time_memory)
            
        def on_step_end(self, args, state, control, **kwargs):
            state.end_memory = torch.cuda.memory_reserved()
            self.average_statistic_memory =  (self.average_statistic_memory* self.record_time_memory +state.end_memory ) / (self.record_time_memory+1)  
            self.record_time_memory +=1
            if self.record_time_memory%50==0:
                print(f"Step {state.global_step}: {self.average_statistic_memory / (1024 ** 3):.2f} GB GPU memory used")
                
    
    if config.system_evaluate =="True":
        trainer.add_callback(GPUTimeCallback())
        trainer.add_callback(GPUMemoryCallback())
    
    trainer.train()
    print(args.output_dir)
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":

    train()
