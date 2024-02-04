import json
import tqdm
import random
import fasttext
import numpy as np
from typing import List
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

random.seed(20)

def count_word(datafiles: List[str], tokenzier: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenzier)
    # uncomment: when padding token is not set, like Mistral 
    tokenizer.pad_token = tokenizer.eos_token

    word_count = dict()
    for f in tqdm.tqdm(datafiles):
        dname = f.split('/')[-2]
        word_count[dname] = {
            'q_len': 0,
            'p_len': 0,
            's_cnt': 0
        }
        for l in tqdm.tqdm(open(f, 'r').readlines()):
            l = json.loads(l)
            q_len = len(tokenizer.tokenize(l['question']))
            p_len = len(tokenizer.tokenize(l['response']))

            word_count[dname]['q_len'] += q_len
            word_count[dname]['p_len'] += p_len
            word_count[dname]['s_cnt'] += 1

    print(json.dumps(word_count, indent=4))

    for dname in word_count.keys():
        word_count[dname]['q_len'] = word_count[dname]['q_len'] / word_count[dname]['s_cnt']
        word_count[dname]['p_len'] = word_count[dname]['p_len'] / word_count[dname]['s_cnt']

    print(json.dumps(word_count, indent=4))


def count_language(datafiles: List[str], savefile: str):
    lang_cls_model = fasttext.load_model('/root/chenzhi/workspace/InternEmbedding/resets/lid.176.bin')
    lang_count = dict()
    for f in tqdm.tqdm(datafiles):
        for l in tqdm.tqdm(open(f, 'r').readlines()):
            l = json.loads(l)
            q_lang, _ = lang_cls_model.predict(l['question'].split('\n')[0], k=1)
            q_lang = q_lang[0]
            if q_lang not in lang_count:
                lang_count[q_lang] = 0
            lang_count[q_lang] += 1

    with open(savefile, 'w') as fw:
        json.dump(lang_count, fw, indent=4)

    print(json.dumps(lang_count, indent=4))

# refer to https://www.jb51.net/article/283109.htm
def plot_language_distribution(lang_dist_file: str):
    language_distribution = json.load(open(lang_dist_file))
    en_zh_languages = []
    en_zh_language_cnt = []
    other_languages = []
    other_language_cnt = []
    for l, lc in language_distribution.items():
        l = l.split('__')[-1]
        if l in ['en', 'zh']:
            en_zh_languages.append(l)
            en_zh_language_cnt.append(lc)
        else:
            other_languages.append(l)
            other_language_cnt.append(lc)

    sorted_other_languages = [(l, lc) for l, lc in zip(other_languages, other_language_cnt)]
    sorted_other_languages = sorted(sorted_other_languages, key=lambda x: x[1], reverse=True)
    other_languages = []
    other_language_cnt = []
    for l, lc in sorted_other_languages:
        other_languages.append(l)
        other_language_cnt.append(lc)

    big_pie_labels = en_zh_languages + ['others']
    big_pie_sizes = en_zh_language_cnt + [sum(other_language_cnt)]
    explode = (0, 0, 0.1)

    small_pie_labels = other_languages[:5]
    small_pie_sizes = other_language_cnt[:5]

    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.pie(big_pie_sizes,
            autopct='%1.1f%%',
            startangle=30,
            labels=big_pie_labels,
            explode=explode)
    
    ax2.pie(small_pie_sizes,
            autopct='%1.1f%%',
            startangle=30,
            labels=small_pie_labels,
            radius=0.5,
            shadow=False)
    
    theta1, theta2 = ax1.patches[-1].theta1, ax2.patches[-1].theta2
    center, r = ax1.patches[-1].center, ax1.patches[-1].r

    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = np.sin(np.pi / 180 * theta2) + center[1]
    con1 = ConnectionPatch(
        xyA=(0, 0.5),
        xyB=(x, y),
        coordsA=ax2.transData,
        coordsB=ax1.transData,
        axesA=ax2,
        axesB=ax1
    )

    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = np.sin(np.pi / 180 * theta1) + center[1]
    con2 = ConnectionPatch(
        xyA=(-0.1, -0.49),
        xyB=(x, y),
        coordsA='data',
        coordsB='data',
        axesA=ax2,
        axesB=ax1
    )

    for con in [con1, con2]:
        con.set_color('gray')
        ax2.add_artist(con)
        con.set_linewidth(1)

    fig.subplots_adjust(wspace=0)

    plt.savefig('language_distribution.png')

def plot_whichlayer_doubley(which_layer_file: str):
    which_layer = json.load(open(which_layer_file))
    x_layer_ids = which_layer['layer_ids']
    y_bank77 = which_layer['Banking77Classification']
    y_askubuntu = which_layer['AskUbuntuDupQuestions']

    fig, ax1 = plt.subplots(figsize=(12,9))
    # title = ('The Number of Players Drafted and Average Career WS/48\nfor each Draft (1966-2014)')
    # plt.title(title,fontsize=20)
    plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    plt.tick_params(axis='both', labelsize=14)
    plot1 = ax1.plot(x_layer_ids, y_bank77, 'b', marker='+', markersize=18, label='Banking77Classification')
    ax1.set_ylabel('ACC of Banking77Classification', fontsize = 18)
    ax1.set_ylim(65, 76)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')    
    ax2 = ax1.twinx()
    plot2 = ax2.plot(x_layer_ids, y_askubuntu, 'g', marker='*', markersize=18, label='AskUbuntuDupQuestions')
    ax2.set_ylabel('MAP of AskUbuntuDupQuestions',fontsize=18)
    ax2.set_ylim(49, 52)
    ax2.tick_params(axis='y', labelsize=14)
    for tl in ax2.get_yticklabels():
        tl.set_color('g')                    
    # ax2.set_xlim(-20, -1)
    ax2.set_xticks(x_layer_ids)
    lines = plot1 + plot2           
    ax1.legend(lines,[l.get_label() for l in lines])    
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0],ax1.get_ybound()[1],9)) 
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0],ax2.get_ybound()[1],9)) 
    for ax in [ax1,ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)                       
    # fig.text(0.1,0.02,'The original content: http://savvastjortjoglou.com/nba-draft-part02-visualizing.html\nPorter: MingYan',fontsize=10)
    plt.savefig("which_layer_mistral.png")

def plot_mrl_doubley(mrl_file: str):
    mrl_dim = json.load(open(mrl_file))
    x_layer_ids = mrl_dim['dimension']
    y_bank77 = mrl_dim['Banking77Classification']
    y_askubuntu = mrl_dim['AskUbuntuDupQuestions']

    fig, ax1 = plt.subplots(figsize=(12,9))
    # title = ('The Number of Players Drafted and Average Career WS/48\nfor each Draft (1966-2014)')
    # plt.title(title,fontsize=20)
    plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    plt.tick_params(axis='both', labelsize=14)
    plot1 = ax1.plot(x_layer_ids, y_bank77, 'b', marker='+', markersize=18, label='Banking77Classification')
    ax1.set_ylabel('ACC of Banking77Classification', fontsize = 18)
    ax1.set_ylim(40, 85)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')    
    ax2 = ax1.twinx()
    plot2 = ax2.plot(x_layer_ids, y_askubuntu, 'g', marker='*', markersize=18, label='AskUbuntuDupQuestions')
    ax2.set_ylabel('MAP of AskUbuntuDupQuestions',fontsize=18)
    ax2.set_ylim(52, 60)
    ax2.tick_params(axis='y', labelsize=14)
    for tl in ax2.get_yticklabels():
        tl.set_color('g')                    
    # ax2.set_xlim(-20, -1)
    ax2.set_xticks(x_layer_ids)
    lines = plot1 + plot2           
    ax1.legend(lines,[l.get_label() for l in lines])    
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0],ax1.get_ybound()[1],9)) 
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0],ax2.get_ybound()[1],9)) 
    for ax in [ax1,ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)                       
    # fig.text(0.1,0.02,'The original content: http://savvastjortjoglou.com/nba-draft-part02-visualizing.html\nPorter: MingYan',fontsize=10)
    plt.savefig("msmarco_ins_mrl.png")


if __name__ == '__main__':
    datafiles = [
        '/fs-computility/llm/chenzhi/datasets_processed/ELI5/train.jsonl', 
        '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl'
    ]
    backbone = "/fs-computility/llm/chenzhi/huggingface_cache/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e"
    # count_word(datafiles, backbone)
    # savefile = 'results/statistics/language_count.json'
    # # count_language(datafiles, savefile)
    # plot_language_distribution(savefile)

    # savefile = 'results/statistics/which_layer.json'
    # plot_whichlayer_doubley(savefile)

    savefile = 'results/statistics/msmarco_ins_mrl.json'
    plot_mrl_doubley(savefile)