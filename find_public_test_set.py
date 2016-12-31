import numpy as np

test_samples = 138
middle = test_samples/2

with open('half_best_sub.csv','w') as output_file:
    with open('best_sub.csv','r') as f:
        all_lines = f.readlines()
        output_file.write(all_lines[0])
        lines = all_lines[1:]
        for i, line in enumerate(lines):
            tokens = line.strip().split(',')
            # print(tokens)
            if int(tokens[1]) >= middle:
                output_file.write(line)
            else:
                output_file.write("{},{},{},{}\n".format(tokens[0], tokens[1], tokens[2], "True"))
