from transformers import pipeline

unmasker = pipeline('fill-mask', model="/data/zhanghy/P-tuning-v2/local_models/roberta-large")
print(unmasker("Hello I'm a <mask> model."))
