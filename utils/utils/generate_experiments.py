import os
import re
import argparse
from itertools import product
from shutil import copyfile


def extract(chunk):
    chunk = chunk[len('<START>'):-len('<STOP>')]
    return chunk.split('<NEXT>')

def next_tag(string):
    tags = ['<start>', '<next>', '<stop>']
    locs = [(string.find(tag), tag) for tag in tags if string.find(tag) != -1]
    if len(locs) == 0:
        return None
    return min(locs)

def can_split(string):
    tags = ['<start>', '<next>', '<stop>']
    locs = [(string.find(tag), tag) for tag in tags if string.find(tag) != -1]
    return len(locs) > 0

def do_split(string):
    tags = ['<start>', '<next>', '<stop>']
    locs = [(string.find(tag), tag) for tag in tags if string.find(tag) != -1]
    tag = min(locs)
    return string[:tag[0]], tag[1], string[tag[0] + len(tag[1]):]

def cross(xs, ys):
    cross = []
    for x in xs:
        for y in ys:
            cross.append(x + y)
    return cross

def process(string):
    after = string
    processed = ['']
    while can_split(after):
        before, tag, after = do_split(after)
        processed = cross(processed, [before])
        assert tag == '<start>'
        depth = 0
        done = False
        strings = ['']
        while not done:
            before, tag, after = do_split(after)
            if tag == '<start>':
                strings[-1] += before + tag
                depth += 1
            elif tag == '<next>' and depth > 0:
                strings[-1] += before + tag
            elif tag == '<next>' and depth == 0:
                strings[-1] += before
                strings.append('')
            elif tag == '<stop>' and depth > 0:
                strings[-1] += before + tag
                depth -= 1
            elif tag == '<stop>' and depth == 0:
                strings[-1] += before
                done = True
            else:
                assert False
        additions = []
        for new_string in strings:
            additions += process(new_string)
        processed = cross(processed, additions)
    processed = cross(processed, [after])
    return processed





def generate_experimets(folder):
    with open(folder + 'set_conf.yaml', 'r') as f:
        conf = f.read()

    starts = [m.start() for m in re.finditer('<START>', conf)]
    stops = [m.end() for m in re.finditer('<STOP>', conf)]
    potential_values = []
    for (start, stop) in zip(starts, stops):
        potential_values.append(extract(conf[start:stop]))

    exp_num = 0
    for values in product(*potential_values):
        version = conf[0:starts[0]]
        for i in range(len(values)):
            if i > 0:
                version += conf[stops[i - 1]:starts[i]]
            version += values[i]
        version += conf[stops[-1]:]
        exp_folder = folder + str(exp_num).zfill(3)
        os.makedirs(exp_folder)
        with open(exp_folder + '/conf.yaml', 'w') as f:
            f.write(version)
        exp_num += 1

    print('Generated \x1b[0;34;40m ' + str(exp_num) + ' \x1b[0m experiments.')

def new_generate_experimets(folder):
    with open(folder + 'set_conf.yaml', 'r') as f:
        conf = f.read()

    processed = process(conf)
    exp_num = 0
    for p in processed:
        exp_folder = folder + str(exp_num).zfill(3)
        os.makedirs(exp_folder)
        with open(exp_folder + '/conf.yaml', 'w') as f:
            f.write(p)
        exp_num += 1
    print('Generated \033[32m' + str(exp_num) + '\033[0m experiments.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--f')
    parser.add_argument('--n')
    parser.add_argument('--copy', action='store_true')
    parser.add_argument('--start', type=int)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--new_id', type=int)
    parser.add_argument('--folder')

    args = parser.parse_args()
    if args.copy:
        start = args.start
        stop = args.stop
        folder = args.folder
        potential_exps = os.listdir(folder)
        exps_by_id = {}
        for potential_exp in potential_exps:
            if os.path.isdir(folder + potential_exp):
                try:
                    exps_by_id[int(potential_exp[:3])] = potential_exp
                except:
                    pass
        #new_id = max([key for key in exps_by_id.keys()]) + 1
        new_id = args.new_id
        for exp_id in range(start, stop + 1):
            new_path = str(new_id).zfill(3) + exps_by_id[exp_id][3:] + '/'
            print(exps_by_id[exp_id], 'into', new_path)
            os.makedirs(folder + new_path)
            copyfile(folder + exps_by_id[exp_id] + '/set_conf.yaml', folder + new_path + 'set_conf.yaml')
            generate_experimets(folder + new_path)
            new_id += 1
    elif args.n:
        new_generate_experimets(args.n)
    else:
        generate_experimets(args.f)
