from bert_serving.client import BertClient

bc = BertClient(ip='localhost')

while True:
    print ('Please input a sentence:')
    sent = input()
    print (bc.encode([sent]))

	


