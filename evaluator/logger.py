import os
import re
import time
import sys

def parseGPUMem(str_content):
    lines = str_content.split("\n")
    target_line = lines[8]
    mem_part = target_line.split("|")[2]
    use_mem = mem_part.split("/")[0]
    total_mem = mem_part.split("/")[1]
    use_mem_int = int(re.sub("\D", "", use_mem))
    total_mem_int = int(re.sub("\D", "", total_mem))
    return use_mem_int, total_mem_int

def parseGPUUseage(str_content):
    lines = str_content.split("\n")
    target_line = lines[8]
    print(target_line)
    useage_part = int(target_line.split("|")[3].split("%")[0])
    return useage_part


def parseProcessMem(str_content, process_name):
    part = str_content.split("|  GPU       PID   Type   Process name                             Usage      |")[1]
    lines = part.split("\n")
    for i in range(len(lines)):
        line = lines[i]
        if line.__contains__(process_name):
            mem_use = int(line[-10:-5])
            return mem_use


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Please input process name to monitor.\nExample: python GPULogger.py target_name")
        exit()
    
    str_command = "nvidia-smi"  
    process_name = sys.argv[1]  
    if len(sys.argv) == 3:
        out_path = sys.argv[2]
    else:
        out_path = "GPU_stat_"+process_name+".txt"
    if len(sys.argv) == 4:
        time_interval = float(sys.argv[3])
    else:
        time_interval = 0.5

    fout = open(out_path, "w")
    fout.write("Timestamp\tGPU Usage Percentage\tGPU Total Mem Usage\tGPU Total Mem Usage Percentage\tProcess Mem Usage\n")

    peak_gpu_mem = 0
    peak_cpu_mem = 0

    process = psutil.Process()
    start_time = time.time()
    
    try:
        while True:
            out = os.popen(str_command)
            text_content = out.read()
            out.close()

            usage_percentage = parseGPUUseage(text_content)
            use_mem, total_mem = parseGPUMem(text_content)
            mem_use = parseProcessMem(text_content, process_name)
            use_percent = round(use_mem * 100.0 / total_mem, 2)

            # Track peak GPU memory usage
            if use_mem > peak_gpu_mem:
                peak_gpu_mem = use_mem

            # Track peak CPU memory usage
            current_cpu_mem = process.memory_info().rss
            if current_cpu_mem > peak_cpu_mem:
                peak_cpu_mem = current_cpu_mem

            str_outline = str(time.time()) + "\t" + str(usage_percentage) + "\t" + str(use_mem) + "\t" + str(use_percent) + "\t" + str(mem_use)
            fout.write(str_outline + "\n")
            print(str_outline + "\t\tPress Ctrl + C to interrupt.")
            time.sleep(time_interval)

    except KeyboardInterrupt:
        end_time = time.time()
        running_time = end_time - start_time
        fout.close()