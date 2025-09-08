# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
from num2words import num2words
import numpy as np
from data.number_tokenizer.numtok import NumTok as nt


def find_num(text):
    res = nt.find_numbers(text)
    # postprocess
    for idx, _ in enumerate(res):
        if any(_[0].endswith(punct) for punct in ',.'):
            res[idx] = (_[0][:-1], _[1], _[2]-1)
    return res


def convert(q, convert_f):
    # try:
    #     return _convert(q, convert_f)
    # except BaseException as err:
    #     print(f'Error while converting! Error message: {err}')
    #     return None
    return _convert(q, convert_f)


def _convert(q, convert_f):

    # replace body numbers with conversions
    def perturb(component):
        # locate numbers in body and question
        number_idx = {_[0]: _[1] for _ in find_num(component)}
        for operand, loc in sorted(list(number_idx.items()), reverse=True, key=lambda x: x[1]):
            component = component[:loc] + convert_f(operand) + component[loc+len(operand):]
        return component

    perturbed_paras = []
    for para in q['paragraphs']:
        new_para = {key: value for key, value in para.items()}
        new_para['text'] = perturb(para['text'])
        perturbed_paras.append(new_para)

    perturbed_ques = []
    for que in q['questions']:
        new_que = {key: value for key, value in que.items()}
        new_que['question'] = perturb(que['question'])
        perturbed_ques.append(new_que)

    new_q = {'table': q['table'],
             'paragraphs': perturbed_paras,
             'questions': perturbed_ques
             }
    return new_q


def convert_noise(q):
    return convert(q, lambda x: x + '.' + str(random.choice(range(10))))


def convert_distri(q):
    return convert(q, lambda x: str(int(x) + int(np.random.normal(1000, 300))))


def convert_lang(q):
    def to_words(num):
        try:
            return num2words(num)
        except:
            return num
    return convert(q, to_words)


def convert_type(q):
    return convert(q, lambda x: x + '.0')


def convert_verbosity(q):
    def to_words(num):
        wrong_value = int(np.random.normal(100, 30))
        return f'{num} (not {wrong_value})'
    return convert(q, to_words)

