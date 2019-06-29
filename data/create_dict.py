import os
from bert_serving.client import BertClient

bc = BertClient(ip='localhost')

dict_file = 'dict'
output_file = 'pretrained'
batch_size = 1000

punctuations = [',', '.', '。', '!', '?', ':', ';', '(', ')']

def create_dict(dict_file, output_file):

    
    if os.path.exists(output_file):
        print (f'Dictionary file exists at {output_file}')
        print ('Continue and overwrite it?(y/n)')
        ans = input().strip()
        if ans != 'y': return 

           
    with open(dict_file, 'r') as f:

        dict_str = ''.join(f.readlines())
        words = dict_str.split()
        words += punctuations
        print (words[-10:])
        input()

        vectors = []

        l = len(words)

        print ('Fetching vectors from bert-service...')

        for si in range(0, l, batch_size):

            ei = si + batch_size
            ei = l if ei > l else ei

            vs = bc.encode(words[si:ei])

            for v in vs:
                vectors.append([x for x in v])

        # print (vectors)

        lines = []
        for w, v in zip(words, vectors):
            v_str = ' '.join([str(x) for x in v])
            lines.append(f'{w} {v_str}\n')

        print (len(lines))
    
        # Write to file
        with open(output_file, 'w') as f:
            print ('Writing to file...', end='')
            f.writelines(lines)
            print ('finished!')
 

if __name__ == '__main__':

    create_dict(dict_file, output_file)

    # Read pretrained file example
    word_to_vec = {}

    with open(output_file, 'r') as f:
        for line in f:
            word_and_vec = line.split()
            word = word_and_vec[0]
            
            vec = [float(s) for s in word_and_vec[1:]]
            word_to_vec[word] = vec

    while True: 
        print ('Please input a Chinese character:')
        word = input().strip()
        print (word_to_vec[word])
        print (len(word_to_vec[word]))
        

            


        





