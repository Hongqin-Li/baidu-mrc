from bert_serving.client import BertClient

bc = BertClient(ip='localhost')

while True:
    print ('Please input a sentence:')
    sents = input()
    sents = [c for c in sents]

    print (sents)
    vec = bc.encode(sents, show_tokens=True)

    print (vec)
    print (vec[0].shape)

	


