import matplotlib.pyplot as plt
from pylab import *       
from adjustText import adjust_text

def autolabel(rects):
    texts = []
    for rect in rects:
        height = rect.get_height()
        text = plt.text(rect.get_x()+rect.get_width()/2., height, '%s' % float(height), size='small')
        texts.append(text)

    return texts

if __name__ == '__main__': 
    # hotpotQA
    # react=[39.7, 30.7, 21.9, 15.8, 3.2]
    # sc=[39.7, 35.8, 25.0, 15.8, 12.6]
    # indv=[43.1, 29.4, 28.1, 15.8, 14.7]
    # # eev=[37.9, 31.8, 25.0, 15.8, 10.5]
    # eevsc=[43.1, 38.2, 37.5, 26.3, 16.8]

    # GSM8K
    react=[46.3, 29.1, 33.3, 23.5, 4.4]
    sc=[52.9, 30.0, 43.3, 35.3, 24.4]
    indv=[42.4, 29.1, 20.0, 17.6, 20.0]
    # eev=[37.9, 31.8, 25.0, 15.8, 10.5]
    eevsc=[57.6, 50.9, 46.7, 47.1, 40.0]

    # name=['ReAct','Self-Consistency','IndV','EEV(+SC)']
    name = ['2','3','4','5','6+']    
    total_width, n = 0.8, 4  
    width = total_width / n 
    x=[0,1,2,3,4]
    a=plt.bar(x, react, width=width, label='ReAct')  
    for i in range(len(x)):  
        x[i] = x[i] + width
    b=plt.bar(x, sc, width=width, label='SC')

    for i in range(len(x)):  
        x[i] = x[i] + width
    c=plt.bar(x, indv, width=width, label='IndV')

    for i in range(len(x)):  
        x[i] = x[i] + width  
    d=plt.bar(x, eevsc, width=width, label='EEV(+SC)', tick_label = name)   
    atexts = autolabel(a)
    btexts = autolabel(b)
    ctexts = autolabel(c)
    dtexts = autolabel(d)
    texts = []
    texts.extend(atexts)
    texts.extend(btexts)
    texts.extend(ctexts)
    texts.extend(dtexts)

    adjust_text(texts,)

    plt.xlabel('#Length of Reasoning')
    plt.ylabel('GSM8K (EM%)')
    # plt.title('学生成绩')
    plt.legend()
    plt.savefig('gsm8k.png')