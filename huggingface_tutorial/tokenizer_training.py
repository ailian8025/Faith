from pathlib import Path

from datasets import load_dataset, DownloadConfig

"""

https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
"""

import os

os.environ["http_proxy"] = "http://localhost:1080"
os.environ["https_proxy"] = "http://localhost:1080"

if __name__ == '__main__':
    # Q: In China there is problem to download the model file/dataset, how to solve this problem

    # A:
    # first you need a proxy like shadowsocket which connect to your outwall computer which I can't help.
    # Then, at the begining I try to setting proxies like this, but at latest version datasets, it doesn't work because it change it's source for this.
    # the solve method, I choose set the global proxy setting like line[4:5]

    proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}
    download_config = DownloadConfig(proxies=proxies)
    dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train", download_config=download_config)
    batch_size = 1000
    all_texts = [dataset[i: i + batch_size]["text"] for i in range(0, len(dataset), batch_size)]

    print(dataset[:5])


    def batch_iterator():
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]


    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", mirror='tuna')
    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=25000)

    print(new_tokenizer(dataset[:5]["text"]))
    new_tokenizer.save_pretrained("my-new-tokenizer")

    tok = new_tokenizer.from_pretrained("my-new-tokenizer")

    # Building tokenizer from scratch
    from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

    tokenizer = Tokenizer(models.WordPiece(unl_token="[UNK]"))

    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    print(tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!"))
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    print(cls_token_id, sep_token_id)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )

    encoding = tokenizer.encode("This is one sentence.", "With this one we have a pair.")
    print(encoding.tokens)
    print(encoding.type_ids)
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    if not os.path.exists("my-new-tokenizer-bert"):
        os.mkdir("my-new-tokenizer-bert")
    path = os.path.join(os.getcwd(), "my-new-tokenizer-bert")
    tokenizer.save(os.path.join(path, "tokenizer.json"))
    tokenizer.model.save(path)

    from transformers import BertTokenizerFast

    new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
