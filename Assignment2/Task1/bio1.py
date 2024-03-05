import json
import re

def convert_multiple_spaces_to_single_space(input_string):
    # Using regular expression to replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', input_string)

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
    currcount=0

    for i in values:
        i=i['value']
        start=i['start']
        end=i['end']
        s=text[start:end+1].split()
        label=i['labels'][0]
        while(curr<len(words)):
            # if ' '.join(s) in ' '.join(words[curr:curr+len(s)]):
            if currcount+len(words[curr])>=start:
                labels[curr]="B_"+label
                currcount+=len(words[curr])+1
                curr+=1
                
                while(currcount<end):
                # for j in range(len(s)-1):
                    labels[curr]='I_'+label
                    currcount+=len(words[curr])+1
                    curr+=1
                break
            currcount+=len(words[curr])+1
            curr+=1
    i=0
    while(i<len(words)):
        if words[i]=="":
            words.pop(i)
            labels.pop(i)
        i+=1
    # print(words)
    return (" ".join(words),labels)

def file_1():
    input_file="NER_TRAIN_JUDGEMENT.json"
    f = open(input_file,)
    input=json.load(f)
    data={}
    for i in input:
        values=i['annotations'][0]['result']
        text=i['data']['text']
        words,labels=bio_tagging_1(values,text)
        data[i['id']]={"text":words,"labels":labels}
    
    file_path = "processed/NER_train.json"
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
        words,labels=bio_tagging_2(words,aspects)
        data[count]={"text":words,"labels":labels}
        count+=1
        # break
    file_path = "Laptop_Review_Val_processed.json"
    # # Write data to JSON file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file,indent=3)
        
# file_2()
file_1()