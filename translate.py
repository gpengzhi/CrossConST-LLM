import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fin', required=True)
    parser.add_argument('--fout', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--src', required=True)
    parser.add_argument('--tgt', required=True)
    parser.add_argument('--prompt', required=True)
    return parser


parser = get_parser()
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained("ALMA-13B-Pretrain", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, args.ckpt)
model.eval()
tokenizer = LlamaTokenizer.from_pretrained("ALMA-13B-Pretrain", padding_side='left')

file_out = open(args.fout, "w")

for line in open(args.fin, 'r').readlines():

    if args.prompt == "gpt-mt":
        prompt = "Translate this from {} to {}:\n{}: {}\n{}:".format(args.src, args.tgt, args.src, line.strip(), args.tgt)
    if args.prompt == "t-enc":
        prompt = "{}: {}\n".format(args.tgt, line.strip())
    if args.prompt == "t-dec":
        prompt = "{}\n{}:".format(line.strip(), args.tgt)
    if args.prompt == "s-enc-t-enc":
        prompt = "{} {}: {}\n".format(args.src, args.tgt, line.strip())
    if args.prompt == "s-enc-t-dec":
        prompt = "{}: {}\n{}:".format(args.src, line.strip(), args.tgt)

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=256, truncation=True).input_ids.cuda()
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, num_beams=5, do_sample=False, max_new_tokens=256)
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    translation = outputs[0][len(prompt):]
    file_out.write(translation.replace("\n", " ") + "\n")
