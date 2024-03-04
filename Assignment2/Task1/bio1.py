import json


def bio_tagging_2(words,aspects):
    labels=['O' for i in range(len(words))]
    curr=0
    for i in aspects:
        s=i['term']
        while(curr<len(words)):
            if words[curr:curr+len(s)]==s:
                labels[curr]="B"
                curr+=1
                for j in range(len(s)-1):
                    labels[curr]='I'
                    curr+=1
                break
            curr+=1
    return labels

def bio_tagging_1(values,text):
    words=text.split(" ")
    labels=['O' for i in range(len(words))]
    curr=0
    for i in values:
        i=i['value']
        start=i['start']
        end=i['end']
        s=text[start:end+1].split()
        label=i['labels'][0]
        while(curr<len(words)):
            if words[curr:curr+len(s)]==s:
                labels[curr]="B_"+label
                curr+=1
                for j in range(len(s)-1):
                    labels[curr]='I'+label
                    curr+=1
                break
            curr+=1
    return labels

def file_1():
    input_file="NER_TEST_JUDGEMENT.json"
    f = open(input_file,)
    input=json.load(f)
    data={}
    for i in input:
        values=i['annotations'][0]['result']
        text=i['data']['text']
        data[i['id']]={"text":text,"labels":bio_tagging_1(values,text)}
    
    file_path = "NER_TEST_JUDGEMENT_processed.json"
    with open(file_path, "w") as json_file:
        json.dump(data, json_file,indent=3)

def file_2():
    input_file="Laptop_Review_Val.json"
    f = open(input_file,)
    input=json.load(f)
    data={}
    count=0
    for i in input:
        words=i['words']
        aspects=i['aspects']
        text=i['raw_words']
        # print(bio_tagging_2(words,aspects))
        data[count]={"text":text,"labels":bio_tagging_2(words,aspects)}
        count+=1
        # break
    file_path = "Laptop_Review_Val_processed.json"
    # # Write data to JSON file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file,indent=3)
        
# file_2()
file_1()