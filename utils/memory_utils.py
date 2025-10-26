import subprocess


def get_gpu_memory_map():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                            capture_output=True)
    output = result.stdout.decode('utf-8')
    memory_maps = []
    for line in output.strip().split('\n'):
        used, total = line.split(',')
        memory_maps.append({'used': int(used), 'total': int(total)})

    for gpu_id, gpu_memory in enumerate(memory_maps):
        print(f"GPU {gpu_id}: Used {gpu_memory['used']}MB, Total {gpu_memory['total']}MB")
