import json

input=json.load(open("NER_TRAIN_JUDGEMENT.json"))
output=json.load(open("processed/NER_train.json"))
def check(labels):
    for i in labels:
        if i!="O":
            return False
    return True
i=0
cnt=0
while(i<len(input)):
    values=input[i]['annotations'][0]['result']
    values2=output[input[i]['id']]['labels']
    if(check(values2) and values==[]):
        cnt+=1
    i+=1
print(cnt)