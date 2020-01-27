def convert_messy_jsonl(file_path):
    jsonified_lines = []
    with open(file_path, 'r') as fp:
        for line in fp:
            jsonified_lines.append(line.replace("'", '"'))
    
    with open(file_path, 'w') as fp:
        fp.writelines(jsonified_lines)