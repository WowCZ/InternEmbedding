import os
import json
import numpy as np
from embedding.eval.metrics import cosine_similarity
from embedding.train.training_embedder import initial_model
from embedding.eval.mteb_eval_wrapper import EvaluatedEmbedder
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def predict_embedder(args):
    embedder, tokenizer = initial_model(args)
    embedder = embedder.to(args.device)
    embedder.eval()

    evaluated_embedder = EvaluatedEmbedder(embedder, tokenizer, args.max_length, args.device)
    
    instructions = ["Given a question, retrieve the corresponding Chinese version of the give question: ", 
                    "Given a question, retrieve the corresponding answer of the give question: ", 
                    "Retrieve semantically similar text: "]
    
    question = "Who is Barack Obama's wife?"
    
    answers = ["奥巴马的妻子是谁？", 
               "Could you tell me the name of Obama's wife?", 
               "Michelle LaVaughn Obama[1] (née Robinson; born January 17, 1964) is an American attorney and author who served as the first lady of the United States from 2009 to 2017, being married to former president Barack Obama.", 
               "Donald Trump, full name Donald John Trump, was born on June 14, 1946. He is an American politician and businessman, and also the 45th President in American history (2017-2021). \Trump was born into a wealthy family in New York City, and his father was a real estate developer. Trump obtained a degree in economics from the Wharton School of Business at the University of Pennsylvania and began working in the real estate and hotel industries. His business empire includes multiple fields such as real estate, hotels, casinos, golf courses, and television programs. \In his political career, Trump successfully ran for president as a Republican candidate in 2016 and implemented a series of policies during his presidency. He pushed for tax cuts and reduced regulatory measures in an attempt to stimulate economic growth. He also promoted immigration policy reforms, including ending the \"Delayed Repatriation of Childhood Arrives in the United States\" (DACA) program and strengthening border security. President Trump's term is also considered a period of controversy and controversy. He has been criticized for his handling of domestic and international affairs during his tenure, including handling the COVID-19 pandemic, foreign policy towards foreign leaders, and addressing issues of race and gender equality within the United States. \At the end of his term, Trump experienced a Capitol Hill riot on January 6, 2021. He gave a speech in front of the Capitol building, encouraging supporters to storm the building to prevent Congress from confirming the results of the 2020 presidential election. This incident led to him becoming the first president in American history to be impeached twice. \Overall, President Trump's presidency is widely regarded as a period full of controversy and controversy, and his policies and measures have had a profound impact on the United States and even the world."]

    results = []
    for instruct in instructions:
        complete_question = instruct + question
        q_embedding = evaluated_embedder.encode([complete_question])
        q_embeddings = q_embedding.expand(len(answers), -1)

        a_embeddings = evaluated_embedder.encode(answers)

        cosine_score = cosine_similarity(q_embeddings.numpy(), a_embeddings.numpy())
        max_answer = np.argmax(cosine_score)

        result = {
            'cosine_score': str(cosine_score),
            'complete_question': complete_question,
            'recalled_answer': answers[max_answer]
        }

        results.append(result)

    result_str = json.dumps(results, ensure_ascii=False, indent=4)
    print(result_str)

    result_file = os.path.join(args.result_dir, f'{args.embedder_name}.json')
    with open(result_file, 'w') as fw:
        fw.write(result_str)

    