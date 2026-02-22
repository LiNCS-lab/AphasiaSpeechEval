# Aphasia Speech Eval

## Overview

**Aphasia Speech Eval** is a Python library developed to evaluate the performance of Automatic Speech Recognition (ASR) systems on noisy conversational speech, particularly from speakers with speech disorders. The library is focused on the analysis of speech with disfluencies such as stuttering, repetitions, and extended pauses, providing metrics to evaluate ASR accuracy and reliability in different scenarios.

## Citation

This library was published as part of the following article:


Dupuis Desroches, J., Ménard, P. A., & Ratté, S. (2026). Evaluating ASR for aphasia: a framework for clinically relevant transcription performance. Aphasiology, 1–26. https://doi.org/10.1080/02687038.2026.2621235


## Features

The library is built around the CHAT format from the Child Language Data Exchange System (CHILDES), widely used within the TalkBank project for representing and analyzing conversational data. This ensures that the library is compatible with established formats and can be easily integrated into existing research workflows. 

- **Session Object**: Convert inputs into a standardized Session object.
- **Benchmarking**: Compare two Session objects to evaluate ASR performance.

## Usage

### Installation

```bash
# Clone the repository
git clone #repo_url

# Change directory
cd aphasiaspeecheval

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
#Linux
source .venv/bin/activate
#Windows
.venv\Scripts\activate

# Install the requirements for development
poetry install --with dev
# Or for running only
poetry install
# or
pip install .
```

### Basic Usage

1. **Import the Library**

```python
import aphasiaspeecheval as ase
```

2. **Create a Session Object**

```python
# From a .Cha file
session = ase.Session.from_chat(file_path='chat_file.cha')

# From a zipped .Cha file with session ID
session = ase.Session.from_zip(file_path='chat_file.zip', session_id='session_id')
```

3. **Benchmarking Two Sessions**

```python
# Provide the reference and predicted sessions
benchmark = ase.Benchmark(reference_session, predicted_session)
# or create directly with file paths to .Cha files
benchmark = ase.Benchmark(reference_file_path, predicted_file_path)

# Calculate the asr performance based on reference tokens/tags vs predicted tokens/tags
print(benchmark.calculate_asr_performance()) # Returns benchmark metrics
#         
#                                             Equal  Substitution  ...  Total  Error Rate
#  corpus_name, session_name TOTAL       CER    7.0           2.0  ...   18.0    0.611111
#                                        WER    1.0           2.0  ...    5.0    0.800000
#                            WORD        CER    4.0           2.0  ...    7.0    0.428571
#                                        WER    0.0           2.0  ...    2.0    1.000000
#                            WORD-NORMAL CER    4.0           2.0  ...    7.0    0.428571
#  .....


#####################################
# Prototype features:

# Print a comparison table of the reference and predicted sessions. Most useful for debugging.
# Does some basic alignment of the reference and predicted timemarks for the purpose of comparison.
print(benchmark.comparison_table)

# Ref Spkr    Reference Utterance    Ref Time          Pred Spkr    Predicted Utterance     Pred Time
# ----------  ---------------------  ----------------  -----------  ----------------------- ----------------
# INV         and how is your re...  (8420, 10310)     PAR          and how is the (..)...  (8420, 10310)
# INV         is it good ?           (10690, 11160)    PAR          is it good ?            (10690, 11160)
# INV         are you happy abou...  (11950, 12790)    PAR          are you happy about...  (11950, 12790)
# INV         no .                   (14355, 14685)    PAR          no                      (14355, 14685)
# ...

# Print the benchmark metrics, according to the predicted tokens/participants/tags
# ie. If the predicted tokens are correct, what is the WER and CER compared to the reference tokens of the same type
print(benchmark) # Prints prettyfied benchmark metrics
print(benchmark.metrics) # Prints the raw benchmark metrics as a dictionary

# Metric                                WER    num_words       CER    num_chars
# -------------------------------  --------  -----------  --------  -----------
# Overall                          0.438691        25134  0.316067       110432
# By Token: PUNCTUATION            0.625365         2988  0.622724         5962
# By Token: WORD-ALL               0.364828        19726  0.259794        96384
# ...
```

4. **Combinining multiple benchmarks**

```python
list_of_benchmarks = [benchmark1, benchmark2, benchmark3]

# Average ASR performance metrics
print(ase.average_asr_performance(list_of_benchmarks))

# Token Type                   CER    Total Characters
# ----------------------  --------  ------------------
# WORD-NORMAL             0.24352                77591
# WORD                    0.246887               77977
# TOTAL                   0.299537              116576
# ...

#####################################
# Prototype features:

# Average metrics, according to the predicted tokens/participants/tags
plot, prettyfied, cer_dataframe, totals_dataframe = ase.average_benchmarks(list_of_benchmarks)
print(prettyfied)

# Metric                                WER    num_words       CER    num_chars
# -------------------------------  --------  -----------  --------  -----------
# Overall                          0.438691        25134  0.316067       110432
# By Token: PUNCTUATION            0.625365         2988  0.622724         5962
# By Token: WORD-ALL               0.364828        19726  0.259794        96384
# ...

```

## Notes on parsing CHAT files

Types.py contains all the possible tags that can be found in the chat file. If the chat file contains a tag that is not in the list, the parser will raise an error. 

## To Do - Next Steps

- [ ] Specialized ASR metrics for some tokens (pauses)

