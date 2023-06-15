from absl import app
from absl import flags

from transformers import AutoModel, AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Define flags
flags.DEFINE_string('dataset', 'squad', 'The name of dataset you are going to use.')
flags.DEFINE_integer('nb_epoch', 16, 'The number of epoch.')

FLAGS = flags.FLAGS


def train_with_info(model, tokenizer, block_size=None):
    if block_size is None:
        block_size = 512

    info_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path='./data/info.txt',
        block_size=block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir='./models/info_trained',
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=32,
        eval_steps=100,
        save_steps=200,
        warmup_steps=500,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=info_dataset,
        eval_dataset=info_dataset,
    )

    trainer.train()


def main(argv):
    del argv    # Unused

    model_checkpoint = 'THUDM/chatglm-6b'

    dataset = load_dataset(FLAGS.dataset)
    model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

    train_with_info(model, tokenizer)

    return 0


if __name__ == '__main__':
    app.run(main)
