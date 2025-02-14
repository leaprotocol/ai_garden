import subprocess
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Construct the vastai CLI command
command = [
    'vastai', 'search', 'offers',
    'external=true', 'verified=false', 'rentable=True', 'gpu_ram>=1', 'dph_total<=0.20',
    '-o', 'dph', '--raw', '--storage', '50'
]

def format_number(num):
    """Formats a number to 3 significant digits."""
    if num is None:
        return "None"
    try:
        float_num = float(num)
        if float_num == 0:
            return "0"
        elif float_num.is_integer():
            return str(int(float_num))
        else:
            return "{:.3g}".format(float_num)
    except (ValueError, TypeError):
        return str(num)

try:
    # Execute the command and capture the output
    vastai_result = subprocess.run(command, capture_output=True, text=True, check=True)
    
    # Parse the JSON output
    offers = json.loads(vastai_result.stdout)
    
    # Prepare data for column formatting
    table_data = []
    header = ["Ask Contract ID", "Machine ID", "Geolocation", "CPU Cores", "CPU GHz", "CPU RAM", "GPU Name", "GPU RAM", "Compute Cap", "DLPerf", "DLPerf/DPH", "Disk BW", "Direct Port Count", "Inet Down", "Inet Up", "Inet Down Cost", "Inet Up Cost", "Storage Cost", "Storage Total Cost", "DPH Total", "DPH Base", "DPH Total Adj", "Total FLOPS", "Verification", "Score", "Min Bid", "Num GPUs", "Rented"]
    table_data.append(header)
    
    for offer in offers:
        row = [
            offer['ask_contract_id'], offer['machine_id'], offer['geolocation'],
            format_number(offer['cpu_cores_effective']), format_number(offer['cpu_ghz']), format_number(offer['cpu_ram']),
            offer['gpu_name'], format_number(offer['gpu_ram']), format_number(offer['compute_cap']),
            format_number(offer['dlperf']), format_number(offer['dlperf_per_dphtotal']), format_number(offer['disk_bw']),
            format_number(offer['direct_port_count']), format_number(offer['inet_down']), format_number(offer['inet_up']),
            format_number(offer['inet_down_cost']), format_number(offer['inet_up_cost']), format_number(offer['storage_cost']),
            format_number(offer['storage_total_cost']), format_number(offer['dph_total']), format_number(offer['dph_base']),
            format_number(offer['dph_total_adj']), format_number(offer['total_flops']), offer['verification'],
            format_number(offer['score']), format_number(offer['min_bid']), format_number(offer['num_gpus']), offer['rented']
        ]
        table_data.append(row)
    
    # Use column command to format the output
    column_command = ['column', '-t', '-s', '\t']
    column_process = subprocess.Popen(column_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    # Pass the table data to column command
    formatted_output, _ = column_process.communicate(input='\n'.join(['\t'.join(map(str, row)) for row in table_data]))
    
    print(formatted_output)

except subprocess.CalledProcessError as e:
    print(f"Error executing vastai command: {e}")
    print(f"Stdout: {e.stdout}")
    print(f"Stderr: {e.stderr}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON output: {e}")
    print(f"Raw output: {vastai_result.stdout}")
