from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from transformers import AutoTokenizer, LlamaConfig

from bci import BCI

def main():

    model_name_or_path = "/n/home07/djimenezbeneto/lab/models/BCI"
    adapter_path = "/n/home07/djimenezbeneto/lab/BCI/peft/"
    merged_path = "/n/home07/djimenezbeneto/lab/BCI/merged/"
    batch_size = 8
    max_length = 64
    lr = 1e-3
    num_epochs = 1
    data_path = ""

    # Prepare dummy data
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Llama was pretrained without a pad token, we have to manually add it for fine-tuning
    tokenizer.add_special_tokens(
        {'pad_token': '[PAD]'}
    )

    input = tokenizer("I am an example", max_length=max_length, return_tensors="pt", truncation=True)


    ## LOAD FROM MERGED
    model = BCI.from_pretrained(merged_path)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    model.eval()
    output = model.generate(input, max_new_tokens=10, do_sample=False)[0]
    tokenizer.decode(output.cpu().squeeze())

    del model

    ## LOAD FROM ADAPTER
    model = BCI.peft_from_adapter(model_name_or_path, adapter_path)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    model.eval()
    output = model.generate(input, max_new_tokens=10, do_sample=False)[0]
    tokenizer.decode(output.cpu().squeeze())



if __name__ == "__main__":
    main()