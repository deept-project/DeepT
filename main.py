
from data import TranslationDataset

if __name__ == "__main__":
    dataset = TranslationDataset('data/train.en', 'data/train.zh')
    print(dataset[0])
