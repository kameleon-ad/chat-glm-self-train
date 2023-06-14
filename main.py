from absl import app

from transformers import AutoModel, AutoTokenizer


def main(_):
    model_checkpoint = 'THUDM/chatglm-6b'

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True).half().cuda()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)
    return 0


if __name__ == '__main__':
    app.run(main)
