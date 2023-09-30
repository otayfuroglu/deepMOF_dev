
file_name = "f1.data"
with open(f"{file_name.split('.')[0]}.xyz", "w") as fl:
    lines = open(file_name).readlines()
    fl.write(f"{len(lines)-5}\n")
    fl.write("from runner data\n")
    for line in lines[2:-3]:
        print(line)
        if line in ["begin", "energy", "charge", "end"]:
            continue
        elements = line.split()
        fl.write(f"{elements[4]} {' '.join(elements[1:4])} {' '.join(elements[5:])}\n")

