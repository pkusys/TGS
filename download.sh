wget -P datasets https://image-net.org/data/tiny-imagenet-200.zip
unzip datasets/tiny-imagenet-200.zip -d datasets/
rm datasets/tiny-imagenet-200.zip

wget -P datasets/bert-config https://huggingface.co/bert-base-uncased/resolve/main/config.json
wget -P datasets/bert-config https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
wget -P datasets/bert-config https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json
wget -P datasets/bert-config https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json
wget -P datasets/bert-config https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt

mkdir datasets/squad-data
wget -P datasets/squad-data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -P datasets/squad-data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json