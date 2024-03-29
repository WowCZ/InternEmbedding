import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from embedding.eval.metrics import cosine_similarity
from embedding.models.modeling_bert import BertEmbedder
from embedding.models.modeling_mistral import MistralEmbedder
from embedding.eval.mteb_eval_wrapper import EvaluatedEmbedder
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def predict_embedder(args):
    mytryoshka_indexes = list(range(args.mytryoshka_size))
    tokenizer = AutoTokenizer.from_pretrained(args.init_backbone)
    if args.backbone_type == 'BERT':
        embedder = BertEmbedder(args.init_backbone, pool_type=args.pool_type, checkpoint_batch_size=64, embed_dim=-1, device=args.device)
    elif args.backbone_type == 'Mistral':
        tokenizer.pad_token = tokenizer.eos_token
        embedder = MistralEmbedder(args.init_backbone, pool_type=args.pool_type, checkpoint_batch_size=32, embed_dim=-1, lora_config=args.peft_lora, which_layer=args.which_layer, mytryoshka_indexes=mytryoshka_indexes).to(args.device)
    
    if os.path.exists(args.embedder_ckpt_path):
        embedder.load_state_dict(torch.load(args.embedder_ckpt_path))

    evaluated_embedder = EvaluatedEmbedder(embedder, tokenizer, args.max_length, args.embedding_norm, args.device)

    # instructions = ["帮忙找到以下中文问题对应的英文版本：", 
    #                 "帮忙找到以下问题对应的正确回复：", 
    #                 "帮忙找到以下中文问题对应的俄文版本：", 
    #                 "帮忙找到问题中和奥巴马相同职业（不包括奥巴马）的人物介绍："]
    
    instructions = ["Help Translate the English version corresponding to the following Chinese question:", 
                    "Help find the correct English response for the following question:", 
                    "Help Translate the Russian version corresponding to the following Chinese question:"]
    
    question = "奥巴马的妻子是谁？"
    
    answers = ["Who is Barack Obama's wife?", 
               "Какой является жена Барака Обамы?", 
               "Obama's wife is Michelle Robinson Obama. Michelle Obama is a lawyer, writer, and public speaker, born on January 17, 1963, and married Obama in 1992. She served at the University of Chicago and during Obama's presidency, as the First Lady, she actively promoted many public policies, including healthy eating and sports, as well as education reform. Michelle Obama is also an active philanthropist, with many of her work focused on improving the health and educational opportunities of children and adolescents. Barack Obama, full name Barack Hussein Obama II, born on August 4, 1961, is an American politician and lawyer. He was the first African American president in American history and served from 2009 to 2017. \Obama was born in Hawaii. His father is Kenyan and his mother is Kansas. He spent his childhood in Hawaii and Indonesia, and returned to Hawaii in the 1970s to continue his studies. He obtained law degrees from Columbia University and Harvard University, and began his career as a professor of constitutional law at the University of Chicago. He later served in the Illinois Senate and the Federal Senate. \In the 2008 presidential election, Obama successfully ran as a Democratic candidate, defeating Republican candidate John McCain. During his tenure, he promoted several major policy reforms, including the Affordable Care Act, also known as the Obama healthcare reform, which aimed to expand healthcare coverage. He also pushed for the Dodd Frank Wall Street Reform and Consumer Protection Act to address the 2008 financial crisis and strengthen consumer protection. \Obama's presidency is also considered an important period for advancing LGBTQ+rights. In 2015, he announced his support for same-sex marriage and promoted the Employment Non Discrimination Act during his tenure, aimed at protecting the LGBTQ+community from employment discrimination. \At the end of his second term, Obama achieved certain achievements in domestic and foreign policies, but also faced some challenges and criticisms. In the years following his term, he continued to be active in public life, promoting various social and political reforms through his foundation, and publishing his memoir A Promised Land. \Overall, Obama's presidency is widely regarded as an important period in promoting social progress and change in the United States, and his policies and measures have had a profound impact on the United States and even the world.", 
               "Donald Trump, full name Donald John Trump, was born on June 14, 1946. He is an American politician and businessman, and also the 45th President in American history (2017-2021). \Trump was born into a wealthy family in New York City, and his father was a real estate developer. Trump obtained a degree in economics from the Wharton School of Business at the University of Pennsylvania and began working in the real estate and hotel industries. His business empire includes multiple fields such as real estate, hotels, casinos, golf courses, and television programs. \In his political career, Trump successfully ran for president as a Republican candidate in 2016 and implemented a series of policies during his presidency. He pushed for tax cuts and reduced regulatory measures in an attempt to stimulate economic growth. He also promoted immigration policy reforms, including ending the \"Delayed Repatriation of Childhood Arrives in the United States\" (DACA) program and strengthening border security. President Trump's term is also considered a period of controversy and controversy. He has been criticized for his handling of domestic and international affairs during his tenure, including handling the COVID-19 pandemic, foreign policy towards foreign leaders, and addressing issues of race and gender equality within the United States. \At the end of his term, Trump experienced a Capitol Hill riot on January 6, 2021. He gave a speech in front of the Capitol building, encouraging supporters to storm the building to prevent Congress from confirming the results of the 2020 presidential election. This incident led to him becoming the first president in American history to be impeached twice. \Overall, President Trump's presidency is widely regarded as a period full of controversy and controversy, and his policies and measures have had a profound impact on the United States and even the world."]
    # a_embeddings = evaluated_embedder.encode(answers)

    for instruct in instructions:
        complete_question = instruct + question
        # i_embedding = evaluated_embedder.encode(instruct)
        q_embedding = evaluated_embedder.encode(complete_question)
        # q_embedding = q_embedding + i_embedding
        q_embeddings = q_embedding.expand(len(answers), -1)

        # ianswers = [instruct+a for a in answers]
        a_embeddings = evaluated_embedder.encode(answers)

        cosine_score = cosine_similarity(q_embeddings.numpy(), a_embeddings.numpy())
        print(cosine_score)
        max_answer = np.argmax(cosine_score)
        print('#'*10, ' Result ', '#'*10)
        print(complete_question)
        print(answers[max_answer])