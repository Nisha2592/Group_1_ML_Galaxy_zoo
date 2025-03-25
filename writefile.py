import subprocess
# How the function is implemented is inside of ./src/get_files.py
from src.get_files import write_file_list

# Re-writing the file_list.txt with the new size
data_size = 9_000
write_file_list(data_size, overwrite=True)

# Run the script
result = subprocess.run(["bash", "write.sh"])
