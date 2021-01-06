import subprocess
from subprocess import PIPE


# Convert ConLL files to free text
process = subprocess.Popen(["scripts/train_conll_to_free_text.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
process = subprocess.Popen(["scripts/dev_conll_to_free_text.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
process = subprocess.Popen(["scripts/test_conll_to_free_text.sh"], stdout=PIPE, stderr=PIPE)
process.wait()

# See stats
process = subprocess.Popen(["scripts/train_tf_stats.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
for line in process.stdout.readlines():
    print(line)
process = subprocess.Popen(["scripts/dev_tf_stats.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
for line in process.stdout.readlines():
    print(line)
process = subprocess.Popen(["scripts/test_tf_stats.sh"], stdout=PIPE, stderr=PIPE)
process.wait()
for line in process.stdout.readlines():
    print(line)

# Merge free-text files


# Translate merged file


# Align translated file


# Project annotations

"""
# Merge translations
process = subprocess.Popen(["scripts/merge_translations.sh"], stdout=PIPE, stderr=PIPE)
process.wait()

# Merge alignments
process = subprocess.Popen(["scripts/merge_alignment.sh"], stdout=PIPE, stderr=PIPE)
process.wait()

for line in process.stdout.readlines():
    print(line)
"""