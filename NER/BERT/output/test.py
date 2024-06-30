with open("lstm.txt","r") as f:
    with open("lstm_0.txt","w") as f_out:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.replace(",","/")
            line+="/\n"
            f_out.write(line)