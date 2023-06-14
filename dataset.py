from absl import app
from absl import flags

from datasets import load_dataset, Dataset, DatasetDict

flags.DEFINE_string('dataset', 'squad', 'The name of dataset you are going to use.')

FLAGS = flags.FLAGS


def split_qa(dataset: Dataset):
    contexts = dataset['context']
    qad = dataset.remove_columns(['id', 'title', 'context'])
    return list(set(contexts)), qad


def main(argv):
    del argv

    dataset = load_dataset(FLAGS.dataset)
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    train_raw_text, train_qad = split_qa(train_dataset)
    valid_raw_text, valid_qad = split_qa(valid_dataset)

    raw_text = '\n'.join(train_raw_text + valid_raw_text)
    qad = DatasetDict()
    qad['train'] = train_qad
    qad['validation'] = valid_qad
    qad.save_to_disk('./data/qad')

    with open('./data/info.txt', 'w', encoding='utf8') as fp:
        fp.write(raw_text)

    return 0


if __name__ == '__main__':
    app.run(main)
