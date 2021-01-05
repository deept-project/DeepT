
import json

if __name__ == "__main__":
    input_path = "data/translation2019zh_valid.json"
    out_zh = input_path + '.zh'
    out_en = input_path + '.en'
    with open(out_zh, 'w', encoding='utf-8') as fzh:
        with open(out_en, 'w', encoding='utf-8') as fen:
            with open(input_path, 'r', encoding='utf-8') as f:
                for each_line in f:
                    obj = json.loads(each_line)
                    fen.write(obj['english'] + '\n')
                    fzh.write(obj['chinese'] + '\n')

    

