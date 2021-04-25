import sys


if __name__ == '__main__':
    argv = sys.argv
    del argv[0]
    md = argv[0]
    fin = open(md, "rt", encoding='UTF8')
    data = fin.read()
    data = data.replace('$', '$$')
    data = data.replace('$$$$', '$$')
    data = data.replace('{{', '{ {')
    data = data.replace('}}', '} }')
    data = data.replace('[png](', '[png](/assets/img/')
    fin.close()

    fin = open(md, "wt", encoding='UTF8')
    fin.write(data)
    fin.close()
