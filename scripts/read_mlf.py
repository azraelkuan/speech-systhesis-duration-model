import sys

mlf_file = ""
output_dir = ""

if len(sys.argv) < 3:
    print("Usage: python {} {} {}".format(sys.argv[0], "mlf_file", "output_dir"))
    exit(1)
else:
    mlf_file = sys.argv[1]
    output_dir = sys.argv[2]


def get_phone(phone_str):
    if len(phone_str) > 7:
        phone = phone_str.split('+')[0].split('-')[1]
    else:
        phone = phone_str.split('[')[0]
    return phone

current_wav_id = ""

with open(mlf_file, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    last_phone = ""
    last_time = 0
    output = None

    for line in lines:
        if "MLF" in line:
            continue
        if ".lab" in line:
            if output is not None:
                output.close()
            current_wav_id = line.strip().split("/")[-1].split(".")[0]
            output = open(output_dir + "/" + current_wav_id + ".txt", 'w', encoding='utf-8')
            last_phone = ""
            last_time = 0
            continue
        if len(line) > 10:
            tmp = line.strip().split()
            phone = get_phone(tmp[2])

            if last_phone == "":
                last_time = int(tmp[0])
                last_phone = phone

            if last_phone != phone:
                current_time = int(tmp[0])
                phone_time = current_time - last_time
                frames = phone_time // 50000
                # save the phone info
                output.write("{} {}\n".format(last_phone, frames))
                last_time = current_time
                last_phone = phone
            end_time = int(tmp[1])
        else:
            phone_time = end_time - last_time
            frames = phone_time // 50000
            output.write("{} {}\n".format(last_phone, frames))




