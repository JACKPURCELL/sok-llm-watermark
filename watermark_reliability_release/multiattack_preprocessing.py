import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()
lex_len = "40"
input_file_path = args.input_path
output_file_path = args.output_path
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        data = json.loads(line)
        if 'w_wm_output' in data and 'w_wm_output_attacked' in data:
            data['w_wm_output'] = data['w_wm_output_attacked']
            del data['w_wm_output_attacked']
        if 'w_wm_output_length' in data and 'w_wm_output_attacked_length' in data:
            data['w_wm_output_length'] = data['w_wm_output_attacked_length']
            del data['w_wm_output_attacked_length']
        if 'w_wm_output_tokd' in data:
            del data['w_wm_output_tokd']
        if 'dipper_inputs_Lex' + lex_len + '_Order0' in data:
            del data['dipper_inputs_Lex' + lex_len + '_Order0']
        output_file.write(json.dumps(data) + '\n')







        
