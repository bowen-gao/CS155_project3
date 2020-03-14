
with open("../data/shakespeare.txt", 'r') as f:
    text = f.read()

lines = text.split('\n')
print(lines)

print('                   8'.strip().isdigit())
newtxt=""
for line in lines:
    if line=='':
        continue
    if line.strip().isdigit():
        continue
    newtxt+=line+'\n'
print(newtxt)