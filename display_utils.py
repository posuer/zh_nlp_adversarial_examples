from IPython.core.display import display, HTML
import numpy as np

from NLImodel.settings import labels_reverse

def html_render(x_orig, x_adv, mark = 'color'):
    x_orig_words = x_orig.split(' ')
    x_adv_words = x_adv.split(' ')
    orig_html = []
    adv_html = []
    # For now, we assume both original and adversarial text have equal lengths.
    assert(len(x_orig_words) == len(x_adv_words))
    for i in range(len(x_orig_words)):
        if x_orig_words[i] == x_adv_words[i]:
            orig_html.append(x_orig_words[i])
            adv_html.append(x_adv_words[i])
        else:
            if mark == 'color':
                orig_html.append(format("<b style='color:green'>%s</b>" %x_orig_words[i]))
                adv_html.append(format("<b style='color:red'>%s</b>" %x_adv_words[i]))
            elif mark == 'square brackets':
                orig_html.append(format("【 %s 】" %x_orig_words[i]))
                adv_html.append(format("【 %s 】" %x_adv_words[i]))
    
    orig_html = ' '.join(orig_html)
    adv_html = ' '.join(adv_html)
    return orig_html, adv_html

def recover_max_vocal(x_orig, x_adv):
    return [x_orig[i] if x_adv[i] == 50000 else x_adv[i] for i in range(len(x_adv))]

def visualize_attack(model, dataset, x_orig, x_adv, x_orig_no_max):
    # remove padding from x_orig_no_max
    orig_list0 = list(x_orig_no_max[0][len(x_orig_no_max[0]) - np.sum(np.sign(x_orig_no_max[0])) : len(x_orig_no_max[0])])
    orig_list1 = list(x_orig_no_max[1][len(x_orig_no_max[1]) - np.sum(np.sign(x_orig_no_max[1])) : len(x_orig_no_max[1])])

    if isinstance(x_adv, np.ndarray):
        
        x_adv_padding = np.pad(x_adv, (len(x_orig[1])-len(x_adv),0), 'constant', constant_values=0)
        adv_pred = model.predict(x_orig[0], x_adv_padding)
        
        #recover max vocabular limit
        adv_list = recover_max_vocal(orig_list1, list(x_adv))
    
    else: #if attack failed
        adv_list = orig_list1

    orig_pred = model.predict(x_orig[0], x_orig[1])
    
    orig_txt0 = dataset.build_text(orig_list0)
    orig_txt1 = dataset.build_text(orig_list1)
    adv_txt = dataset.build_text(adv_list)

    orig_html0 = orig_txt0
    orig_html1, adv_html = html_render(orig_txt1, adv_txt)

    print('Original Prediction = %s. (Confidence = %0.2f) ' %(( labels_reverse[np.argmax(orig_pred)] ), np.max(orig_pred)*100.0))
    print('Premise')
    display(HTML(orig_html0))
    print('Hypothesis')   
    display(HTML(orig_html1))
    print('---------  After attack -------------')
    print('New Prediction = %s. (Confidence = %0.2f) ' %(( labels_reverse[np.argmax(adv_pred)] if isinstance(x_adv, np.ndarray) else 'Failed'  ), (np.max(adv_pred)*100.0 if isinstance(x_adv, np.ndarray) else 0.00 )))

    display(HTML(adv_html))
'''
def save_all_attack(model, dataset, x_orig_list, x_adv_list, file_name = 'attack_result.csv'):
    writer = open(file_name, 'w')
    writer.write('ID,Attacked,Hypothesis,Premise,OrigPred,NewPred,OrigConf,NewConf\n')
    for idx in range(len(x_adv_list)):
        x_orig = x_orig_list[idx]
        x_adv = x_adv_list[idx]

        writer.write(str(idx)+',')
        if not isinstance(x_adv, np.ndarray):
            writer.write('This attack failed, , , , , ,\n')
            continue

        x_adv_padding = np.pad(x_adv, (len(x_orig[1])-len(x_adv),0), 'constant', constant_values=0)

        orig_pred = model.predict(x_orig[0], x_orig[1])
        adv_pred = model.predict(x_orig[0], x_adv_padding)
        
        # remove padding
        orig_list0 = list(x_orig[0][len(x_orig[0]) - np.sum(np.sign(x_orig[0])) : len(x_orig[0])])
        orig_list1 = list(x_orig[1][len(x_orig[1]) - np.sum(np.sign(x_orig[1])) : len(x_orig[1])])
        adv_list = recover_max_vocal(orig_list1, list(x_adv))
        
        orig_txt0 = dataset.build_text(orig_list0)
        orig_txt1 = dataset.build_text(orig_list1)
        adv_txt = dataset.build_text(adv_list)

        orig_html0 = orig_txt0
        orig_html1, adv_html = html_render(orig_txt1, adv_txt, mark = 'square brackets')

        print('Original Prediction = %s. (Confidence = %0.2f) ' %(('Entailment' if np.argmax(orig_pred) == 1 else 'Contradiction'), np.max(orig_pred)*100.0))
        print('Premise')
        display(HTML(orig_html0))
        print('Hypothesis')   
        display(HTML(orig_html1))
        print('---------  After attack -------------')
        print('New Prediction = %s. (Confidence = %0.2f) ' %(('Entailment' if np.argmax(adv_pred) == 1 else 'Contradiction'), np.max(adv_pred)*100.0))

    print('All saved in', file_name)
    writer.close()
'''
def visualize_attack2(dataset, test_idx, x_orig, x_adv, label):
    
    raw_text = dataset.test_text[test_idx]
    print('RAW TEXT: ')
    display(HTML(raw_text))
    print('-'*20)
    x_len = np.sum(np.sign(x_orig))
    orig_list = list(x_orig[:x_len])
    #orig_pred = model.predict(sess,x_orig[np.newaxis,:])
    #adv_pred = model.predict(sess, x_adv[np.newaxis,:])
    orig_txt = dataset.build_text(orig_list)
    if x_adv is None:
        adv_txt = "FAILED"
    else:
        adv_list = list(x_adv[:x_len])
        adv_txt = dataset.build_text(adv_list)
    orig_html, adv_html = html_render(orig_txt, adv_txt)
    print('Original Prediction = %s.  ' %('Positive' if label == 1 else 'Negative'))
    display(HTML(orig_html))
    print('---------  After attack -------------')
    print('New Prediction = %s.' %('Positive' if label == 0 else 'Negative'))

    display(HTML(adv_html))