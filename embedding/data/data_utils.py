dataset_sampling_ratios = {
    'ELI5': 1.0,
    'HotpotQA': 1.0,
    'MSMARCO': 1.0,
    'MultiNLI': 1.0,
    'Quora': 1.0,
    'MIRACL': 1.0,
    'MrTyDi': 1.0,
    'SQuAD': 1.0,
    'NautralQuestions': 1.0,
    'TriviaQA': 1.0,
    'FEVER': 1.0,
    'DuReader': 1.0,
    'T2Ranking': 1.0,
    'MSMARCO_Triple': 1.0,
    'STAllNLI': 1.0,
    'STELI5': 1.0,
    'STGooQA': 1.0,
    'STSpecter': 1.0,
    'STStackexchangeDup': 1.0,
    'STWikiHow': 1.0,
    'STYahooQA': 1.0
}

dataset_task_prompts = {
    'ELI5': [
        'Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum'
    ],
    'HotpotQA': [
        'Given a multi-hop question, retrieve documents that can help answer the question'
    ],
    'MSMARCO': [
        'Given a web search query, retrieve relevant passages that answer the query',
        'Given a web search query, retrieve relevant documents that answer the query'
    ],
    'MultiNLI': [
        'Given a premise, retrieve a hypothesis that is entailed by the premise',
        'Retrieve semantically similar text'
    ],
    'Quora': [
        'Given a question, retrieve questions that are semantically equivalent to the given question',
        'Find questions that have the same meaning as the input question'
    ],
    'MIRACL': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'MrTyDi': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'SQuAD': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'NautralQuestions': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'TriviaQA': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'FEVER': [
        'Given a claim, retrieve documents that support or refute the claim'
    ],
    'DuReader': [
        'Given a Chinese search query, retrieve web passages that answer the question'
    ],
    'T2Ranking': [
        'Given a Chinese search query, retrieve web passages that answer the question'
    ],
    'MSMARCO_Triple': [
        'Given a web search query, retrieve relevant passages that answer the query',
        'Given a web search query, retrieve relevant documents that answer the query'
    ],
    'STELI5': [
        'Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum'
    ],
    'STAllNLI': [
        'Given a premise, retrieve a hypothesis that is entailed by the premise',
        'Retrieve semantically similar text'
    ],
    'STGooQA': [
        'Provided a google question, retrieve the highest voted answers'
    ],
    'STSpecter': [
        'Provided a title of the scientific publication, retrieve the related title of the publication'
    ],
    'STStackexchangeDup': [
        'Given a question, retrieve questions that are semantically equivalent to the given question',
        'Find questions that have the same meaning as the input question'
    ],
    'STWikiHow': [
        'Given a summary, retrieve the corresponding documents'
    ],
    'STYahooQA': [
        'Provided a Yahoo question, retrieve the highest voted answers'
    ]
}

# initial dataloader
## all samples are randomly sampled, where one batch contains different domain samples. 
# training_datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/ELI5/train.jsonl', 
#                   '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl']

## all samples in a batch are sampled from the same task, which we called the in-domain batch sampling. 
# training_datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/InDomain/train.jsonl']

##  msmarco dataset with hard negative sampling
# training_datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/MSMARCO_Triple/train.jsonl']

# training_datatset_files = [
#             '/fs-computility/llm/chenzhi/datasets_processed/ELI5/filtered_phase2_train.jsonl', 
#             '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/Quora/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/FEVER/filtered_phase2_train.jsonl'
# ]

training_datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/STELI5/train.jsonl', 
                '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/train.jsonl',
                '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/train.jsonl']