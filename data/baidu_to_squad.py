import glob
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw-directory', '-r', type=str, help='directory of baidu raw json files to format', default='./data_chunks/')
parser.add_argument('--out-directory', '-o', type=str, help='output directory of SQUAD-like json files', default='./squad_like_datas/')

args = parser.parse_args()

INPUT_DIR = args.directory

filepath = INPUT_DIR + "*.json"
files = glob.glob(filepath)

for f in files:

    datas = []

    with open(f, 'r') as fp:
        try:
            datas = json.load(fp)
            print (f'File "{f}" has been formatted.')
            continue
        except:
            datas = []
            pass

    with open(f, 'r') as fp:
        print (f'File "{f}" unformatted')
        lines = fp.readlines()
        for line in lines:
            data = json.loads(line)
            datas.append(data)
            

    with open(f, 'w') as fp:
        json.dump(datas, fp, ensure_ascii=False, indent=4)
        print (f'Format file {f} successfully.')


