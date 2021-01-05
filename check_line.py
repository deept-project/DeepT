
import tqdm
def count_line(path):
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for each_line in tqdm.tqdm(iterable=f):
            count+=1
    return count

        

if __name__ == "__main__":
    path = 'data/ai_challenger_2017_train'
    left_count = count_line(path+'.en')
    right_count = count_line(path+'.zh')
    print(left_count, right_count)