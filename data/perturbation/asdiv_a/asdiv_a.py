# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
from num2words import num2words
import numpy as np
import sys
sys.path.append('../..')
from data.number_tokenizer.numtok import NumTok as nt


def find_num(text):
    res = nt.find_numbers(text)
    # postprocess
    for idx, _ in enumerate(res):
        if any(_[0].endswith(punct) for punct in ',.'):
            res[idx] = (_[0][:-1], _[1], _[2]-1)
    return res


def get_operands(q):
    eq = q['Formula'].split('=')[0]
    for idx, char in enumerate(eq):
        if char in '+-*/()':
            eq = eq[:idx] + ' ' + eq[idx+1:]
    operands = eq.split()
    return operands


def is_all_operands_int(operands):
    return not any(['.' in operand for operand in operands])


def is_all_operands_in_body(operands, q):
    body_numbers = [_[0] for _ in find_num(q['Body'])]
    return all([str(operand) in body_numbers for operand in operands])


def is_all_operands_unique(operands):
    return len(set(operands)) == len(operands)


def is_not_ceil_floor_division(q):
    return ' ' not in q['Formula']


def body_contains_verbose(operands, q):
    body_numbers = find_num(q['Body'])
    if len(body_numbers) > len(operands):
        return True
    else:
        return False


def contains_no_difference(q):
    return '-' not in q['Formula']


def is_simple(q):
    operands = get_operands(q)
    is_int = is_all_operands_int(operands)
    is_in_body = is_all_operands_in_body(operands, q)
    is_unique = is_all_operands_unique(operands)
    simple_div = is_not_ceil_floor_division(q)
    return is_int and is_in_body and is_unique and simple_div


def is_simple_for_distri(q):
    operands = get_operands(q)
    is_in_body = is_all_operands_in_body(operands, q)
    is_unique = is_all_operands_unique(operands)
    simple_div = is_not_ceil_floor_division(q)
    no_diff = contains_no_difference(q)
    return is_in_body and is_unique and simple_div and no_diff


def convert(q, convert_f, convert_formula=True):
    try:
        return _convert(q, convert_f, convert_formula)
    except BaseException as err:
        print(f'Error while converting! Error message: {err}')
        return None


def _convert(q, convert_f, convert_formula=True):

    body = q['Body']
    formula, answer = q['Formula'].split('=')

    # locate numbers in body and formula
    body_number_idx = {_[0]: _[1] for _ in find_num(body)}
    formula_number_idx = {_[0]: _[1] for _ in find_num(formula)}

    if convert_formula:
        # prepare operand perturbations
        all_operands = get_operands(q)
        conversion = {}
        for operand in all_operands:
            new_operand = convert_f(operand)
            conversion[operand] = new_operand

        # get operand locations in body and formula
        body_locations = []
        formula_locations = []
        for operand in conversion:
            body_locations.append((body_number_idx[operand], operand))
            formula_locations.append((formula_number_idx[operand], operand))

        # build new body and formula
        for loc, operand in sorted(body_locations, reverse=True, key=lambda x: x[0]):
            body = body[:loc] + conversion[operand] + body[loc+len(operand):]
        for loc, operand in sorted(formula_locations, reverse=True, key=lambda x: x[0]):
            formula = formula[:loc] + conversion[operand] + formula[loc+len(operand):]

    else:
        # replace body numbers with conversions
        for operand, loc in sorted(list(body_number_idx.items()), reverse=True, key=lambda x: x[1]):
            body = body[:loc] + convert_f(operand) + body[loc+len(operand):]

    new_answer = str(eval(formula))
    new_formula = formula + '=' + new_answer
    new_q = {'@ID': q['@ID'],
             '@Grade': q['@Grade'],
             '@Source': q['@Source'],
             'Body': body,
             'Question': q['Question'],
             'Solution-Type': q['Solution-Type'],
             'Answer': q['Answer'].replace(answer, new_answer),
             'Formula': new_formula
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
    return convert(q, to_words, convert_formula=False)


def convert_type(q):
    return convert(q, lambda x: x + '.0')


def convert_verbosity(q):
    def to_words(num):
        wrong_value = int(np.random.normal(100, 30))
        return f'{num} (not {wrong_value})'
    return convert(q, to_words, convert_formula=False)
