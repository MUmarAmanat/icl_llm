# Databricks notebook source
# MAGIC %md
# MAGIC ## Install libraries

# COMMAND ----------

# MAGIC %pip install torch torchdata transformers datasets==2.11.0 --quiet

# COMMAND ----------

import random 
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset

# COMMAND ----------

hf_data = "knkarthick/dialogsum"

dataset = load_dataset(hf_data)

# COMMAND ----------

# MAGIC %md 
# MAGIC display few examples

# COMMAND ----------

for ex_num, ex_ind in enumerate(random.sample(range(len(dataset["test"])), 5)):
  print(f"Example#{ex_num + 1}")
  print(f"Input:\n{dataset['test'][ex_ind]['dialogue']}\n")
  print(f"Summary:\n{dataset['test'][ex_ind]['summary']}\n")
  print("\t\t----------------------------\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load pretrained model

# COMMAND ----------

model_name = 'google/flan-t5-large'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# COMMAND ----------

sentence = "Oppenheimer: Now I Am Become Death, the Destroyer of Worlds."
enc_sen = tokenizer(sentence, return_tensors="pt")



# COMMAND ----------

dec_sen = tokenizer.decode(enc_sen["input_ids"][0],
                           skip_special_tokens=True)

print(f"Encoded sentence: {enc_sen['input_ids'][0]}\n")
print(f"Decoded sentence: {dec_sen}\n")

# COMMAND ----------

# MAGIC %md 
# MAGIC Randomly generates model's response

# COMMAND ----------

ex_for_model = random.sample(range(len(dataset["test"])), 2)
ex_to_summarize = [942, 979]

# COMMAND ----------

for ex_num, ex_ind in enumerate(ex_to_summarize):
  input_prompt = dataset["test"][ex_ind]["dialogue"]
  act_summary = dataset["test"][ex_ind]["summary"]

  model_inp = tokenizer(input_prompt, return_tensors="pt")
  model_out = tokenizer.decode(model.generate(model_inp["input_ids"],
                                              max_new_tokens=50)[0],
                               skip_special_tokens=True
                               )
  
  print(f"Example#{ex_num}")
  print(f"Input prompt: {input_prompt}\n")
  print(f"Actual Summary: {act_summary}\n")
  print(f"Model Summary: {model_out}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Zero shot inference

# COMMAND ----------

# MAGIC %md
# MAGIC ### Provide instructions
# MAGIC
# MAGIC Previously we didn't instruct the model what to do. Now we are explicitly instructing the model to summarize the conversation between two people.

# COMMAND ----------

for ex_num, ex_ind in enumerate(ex_to_summarize):
  input_prompt = f"Summarize the following conversation.\n {dataset['test'][ex_ind]['dialogue']}"
  act_summary = dataset["test"][ex_ind]["summary"]

  model_inp = tokenizer(input_prompt, return_tensors="pt")
  model_out = tokenizer.decode(model.generate(model_inp["input_ids"],
                                              max_new_tokens=50)[0],
                               skip_special_tokens=True
                               )
  
  print(f"Example#{ex_num + 1}")
  print(f"Input prompt: {input_prompt}\n")
  print(f"Actual Summary: {act_summary}\n")
  print(f"Model Summary: {model_out}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## One shot Inference

# COMMAND ----------

for ex_num, (ex_m, ex_sum) in enumerate(zip(ex_for_model, ex_to_summarize)):
  input_prompt = f"""Dialogue between two people:
                   {dataset['test'][ex_m]['dialogue']}
                    what they were talking about? 
                    {dataset['test'][ex_m]['summary']}

                    Dialogue between two people:
                    {dataset['test'][ex_sum]['dialogue']}
                    what they were talking about? 
                    """

  model_inp = tokenizer(input_prompt, return_tensors="pt")
  model_out = tokenizer.decode(model.generate(model_inp["input_ids"],
                                              max_new_tokens=50)[0],
                               skip_special_tokens=True
                               )
  
  print(f"Example#{ex_num + 1}")
  print(f"Input prompt: {input_prompt}\n")
  print(f"Actual Summary: {dataset['test'][ex_sum]['summary']}\n")
  print(f"Model Summary: {model_out}\n")
  print("\t\t------------------")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Few Shot Inference

# COMMAND ----------

ex_for_model = random.sample(range(len(dataset["test"])), 5)

# COMMAND ----------

for ex_m in ex_for_model:
  input_prompt += f"""Dialogue between two people: 
                    {dataset['test'][ex_m]['dialogue']}
                    what they were talking about? 
                    {dataset['test'][ex_m]['summary']}

                    """

# COMMAND ----------

for ex_num, ex_sum in enumerate(ex_to_summarize):
  input_prompt += f"""
                    Dialogue between two people:
                    {dataset['test'][ex_sum]['dialogue']}
                    what they were talking about? 
                    """

  model_inp = tokenizer(input_prompt, return_tensors="pt")
  model_out = tokenizer.decode(model.generate(model_inp["input_ids"],
                                              max_new_tokens=50)[0],
                               skip_special_tokens=True
                               )
  
  print(f"Example#{ex_num + 1}")
  print(f"Input prompt: {input_prompt}\n")
  print(f"Actual Summary: {dataset['test'][ex_sum]['summary']}\n")
  print(f"Model Summary: {model_out}\n")
  print("\t\t------------------")

# COMMAND ----------

# MAGIC %md
# MAGIC References
# MAGIC 1. https://www.coursera.org/learn/generative-ai-with-llms
# MAGIC 2. https://aws.amazon.com/blogs/machine-learning/build-a-news-based-real-time-alert-system-with-twitter-amazon-sagemaker-and-hugging-face/
# MAGIC

# COMMAND ----------


