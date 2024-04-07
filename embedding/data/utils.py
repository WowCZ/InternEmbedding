
def get_preference_pseduolabel_0326(sample, extract_pseduo_label=True):
        # import pdb; pdb.set_trace()
        if extract_pseduo_label:
            puyu_label = sample['preds']['ad']['pred_tag']
        else:
            puyu_label = sample['labels']['ad']
        if puyu_label is not None:
            puyu_label = int(puyu_label)
            return puyu_label 
        else:
            return None
    
def tackle_preference_sample_0326(sample, extract_pseduo_label=True):
    if extract_pseduo_label:
        puyu_label = sample['preds']['ad']['pred_tag']
    else:
        puyu_label = sample['labels']['ad']

    text_a = sample['texts']['texts'][0]['content']
    text_b = sample['texts']['texts'][1]['content']
    puyu_label = int(puyu_label)
    assert puyu_label in [0, 1]
    if puyu_label == 1:
        text_a, text_b = text_b, text_a
    return {
        'question': text_a,
        'response': text_b,
        'negative_response': None
    }

def tackle_preference_sample_0407(sample):
    text_a = sample['texts'][0]['content']
    text_b = sample['texts'][1]['content']
    label = sample['preds']['ad']['pred_tag']

    prompt_temp = f"""
Please select the text that ensembles an advertisement more.

# Option 0
```
{text_a}
```

# Option 1
```
{text_b}
```
"""
    prompt = prompt_temp.format(text_a=text_a, text_b=text_b)
    return {
        'question': prompt,
        'response': None,
        'negative_response': None, # either 0 or 1
        'label': label
    }

def tackle_haijun_preference_0321(sample):
    text_a = sample['text_a']
    text_b = sample['text_b']
    label = sample['label']
    assert label in [0, 1]
    if label == 1:
        text_a, text_b = text_b, text_a
    return {
        'question': text_a,
        'response': text_b,
        'negative_response': None
    }
