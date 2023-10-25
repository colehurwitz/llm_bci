from transformers import AutoTokenizer

from bci import BCI

def main():

    model_name_or_path = "/n/home07/djimenezbeneto/lab/models/BCI"
    adapter_path = "/n/home07/djimenezbeneto/lab/BCI/peft/"
    merged_path = "/n/home07/djimenezbeneto/lab/BCI/merged/"
    max_length = 64
    data_path = ""

    

    # Prepare dummy data
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Llama was pretrained without a pad token, we have to manually add it for fine-tuning
    tokenizer.add_special_tokens(
        {'pad_token': '[PAD]'}
    )
    inputs = tokenizer("The number following 1 is ", max_length=max_length, return_tensors="pt", truncation=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    ## LOAD FROM MERGED
    model = BCI.from_pretrained(merged_path, device_map="auto")
    model.decoder.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    model.eval()
    print(model.hf_device_map)
    print(model)
    output = model.generate(**inputs, max_new_tokens=10, do_sample=False,  pad_token_id=tokenizer.pad_token_id)[0]
    print(tokenizer.decode(output.cpu().squeeze()))

    del model

 
    ## LOAD FROM ADAPTER
    model = BCI.peft_from_adapter(model_name_or_path, adapter_path, device_map="auto")
    model.decoder.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    model.eval()
    print(model.hf_device_map)
    print(model)
    output = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)[0]
    print(tokenizer.decode(output.cpu().squeeze()))



if __name__ == "__main__":
    main()