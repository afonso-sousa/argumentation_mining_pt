import subprocess
from subprocess import PIPE


# Convert ConLL files to free text
process = subprocess.Popen(["scripts/train_conll_to_free_text.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
print("Train ConLL to free-text done")
process = subprocess.Popen(["scripts/dev_conll_to_free_text.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
print("Dev ConLL to free-text done")
process = subprocess.Popen(["scripts/test_conll_to_free_text.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
print("Test ConLL to free-text done")

# See stats
process = subprocess.Popen(["scripts/train_ft_stats.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
print("Train free-text stats:")
for line in process.stdout.readlines():
    print(line)
process = subprocess.Popen(["scripts/dev_ft_stats.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
print("Dev free-text stats:")
for line in process.stdout.readlines():
    print(line)
process = subprocess.Popen(["scripts/test_ft_stats.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
print("Test free-text stats:")
for line in process.stdout.readlines():
    print(line)

# Merge free-text files
process = subprocess.Popen(["scripts/merge_ft_all.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
process = subprocess.Popen(["scripts/all_ft_stats.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
print("Merged free-text stats:")
for line in process.stdout.readlines():
    print(line)

# Translate merged file
process = subprocess.Popen(["scripts/translate_merged.sh"], stdout=PIPE, stderr=PIPE)
for line in process.stdout:
    print(line)

# Align translated file
process = subprocess.Popen(["scripts/align_translated.sh"], stdout=PIPE, stderr=PIPE)
process.wait()

# Project annotations
process = subprocess.Popen(["scripts/generate_conll_split.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
