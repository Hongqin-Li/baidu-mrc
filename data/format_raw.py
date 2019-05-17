import glob
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, help='directory of json files to format', default='./data_chunks')

args = parser.parse_args()

INPUT_DIR = args.directory

filepath = INPUT_DIR + "./*.json"
files = glob.glob(filepath)

for f in files:

    datas = []

    with open(f, 'r') as fp:
        try:
            json.load(fp)
            print (f'File "{f}" has been formatted.')
            continue
        except:
            for line in fp:
                data = json.loads(line)
                datas.append(data)
            

    with open(f, 'w') as fp:
        json.dump(datas, fp, ensure_ascii=False)
        print (f'Format file {f} successfully.')


