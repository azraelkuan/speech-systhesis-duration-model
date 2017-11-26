import difflib
import sys


def compare_string(wav_id, ssp, tsp):
    source_path = "../dur/{}_{}.txt".format(ssp.upper(), wav_id)
    target_path = "../dur/{}_{}.txt".format(tsp.upper(), wav_id)
    source_phones_list = []
    target_phones_list = []
    with open(source_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            source_phones_list.append(line.strip().split()[0])
    with open(target_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            target_phones_list.append(line.strip().split()[0])
    source_string = " ".join(source_phones_list)
    target_string = " ".join(target_phones_list)
    print(len(source_phones_list), len(target_phones_list))
    print(source_string)
    print(target_string)
    s = difflib.SequenceMatcher(None, source_string, target_string)
    all_ops = []
    for opcode in s.get_opcodes():
        all_ops.append(opcode)

    for opcode in reversed(all_ops):
        if opcode[0] == 'insert':
            start = opcode[3]
            index = 0
            for i, _ in enumerate(target_string):
                if _ == " ":
                    index += 1
                if i == start:
                    target_phones_list.pop(index)

        if opcode[0] == "delete":
            start = opcode[1]
            index = 0
            for i, _ in enumerate(source_string):
                if _ == " ":
                    index += 1
                if i == start:
                    source_phones_list.pop(index)

    print(source_phones_list)
    print(target_phones_list)




if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python {} {} {} {}".format(sys.argv[0], "wav_id", "source_speaker", "target_speaker"))
    wav_id = sys.argv[1]
    source_speaker = sys.argv[2]
    target_speaker = sys.argv[3]
    compare_string(wav_id, source_speaker, target_speaker)