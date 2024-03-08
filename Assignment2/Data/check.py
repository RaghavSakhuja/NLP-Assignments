import json

input=json.load(open("NER_TEST_JUDGEMENT.json"))
output=json.load(open("processed/NER_test.json"))
def check(labels):
    cnt=0
    for i in labels:
        if "B_" in i:
            cnt+=1
            # return False
    return cnt
i=0
cnt=0
cnt2=0
while(i<len(input)):
    values=input[i]['annotations'][0]['result']
    if input[i]['id'] in output:
        values2=output[input[i]['id']]['labels']
        text=input[i]['data']['text'].split(" ")
        # if(check(values2)and values!=[]):'
        if(check(values2)!=len(values)):
            print(values)
            print()
            print(text)
            print(values2)
            print(len(values),check(values2))
            # break
            cnt+=1
        cnt2+=1
    i+=1
print(cnt,cnt2)