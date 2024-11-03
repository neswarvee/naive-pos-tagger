"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h :      this help message and exit
    -d FILE : use FILE as data to create a new lexicon file
    -t FILE : apply lexicon to test data in FILE
"""
################################################################

import sys, re, getopt
from collections import defaultdict
import json

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:], 'hd:t:')

opts = dict(opts)

def printHelp():
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit()
    
if '-h' in opts:
    printHelp()

if '-d' not in opts:
    print("\n** ERROR: must specify training data file (opt: -d FILE) **", file=sys.stderr)
    printHelp()

if len(args) > 0:
    print("\n** ERROR: unexpected input on commmand line **", file=sys.stderr)
    printHelp()

################################################################
term_postag_count = defaultdict(lambda: defaultdict(int))
input_file = opts['-d'] 
#populate term_postag_count dictionary
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        word_pos_pairs = line.strip().split()
        for pair in word_pos_pairs:
            token, pos_tag= pair.split('/',1)
            term_postag_count[token][pos_tag] += 1

# Step 2: Calculate relative frequencies and sort in descending order
pos_tag_counts = defaultdict(int)
total_tag_count = 0

for pos_dict in term_postag_count.values():
    for pos, count in pos_dict.items():
        pos_tag_counts[pos] += count
        total_tag_count += count

pos_tag_frequencies = {pos: count / total_tag_count for pos, count in pos_tag_counts.items()}
sorted_pos_tags = sorted(pos_tag_frequencies.items(), key=lambda x: x[1], reverse=True)

most_common_tag, most_common_freq = sorted_pos_tags[0] #sorted_pos_tags is a tuple
accuracy_score = most_common_freq

def ambiguous():
    # Step 1: Determine ambiguous and unambiguous types
    total_types = len(term_postag_count)           # Total number of unique words (types)
    ambiguous_types_count = 0                      # Count of ambiguous types (with more than one POS tag)
    unambiguous_token_occurrences = 0              # Count of token occurrences for unambiguous types
    total_token_occurrences = 0                    # Total number of token occurrences

    for token, pos_dict in term_postag_count.items():
        # Count the number of POS tags for this token
        pos_count = len(pos_dict)
        
        # Sum total occurrences for this token across all POS tags
        token_occurrences = sum(pos_dict.values())
        total_token_occurrences += token_occurrences
        
        if pos_count > 1:
            # Token has more than one POS tag -> ambiguous type
            ambiguous_types_count += 1
        else:
            # Token has only one POS tag -> unambiguous
            unambiguous_token_occurrences += token_occurrences

    # Calculate proportions
    proportion_ambiguous_types = ambiguous_types_count / total_types
    proportion_unambiguous_token_occurrences = unambiguous_token_occurrences / total_token_occurrences
    # Print results
    print(f"Proportion of ambiguous types: {proportion_ambiguous_types:.2%}")
    print(f"Proportion of unambiguous token occurrences: {proportion_unambiguous_token_occurrences:.2%}")

#Building the pos tagger:
correctly_tagged_tokens = 0
total_tokens = 0

for token, pos_dict in term_postag_count.items():
    most_common_pos = max(pos_dict, key=pos_dict.get)
    most_common_pos_count = pos_dict[most_common_pos]

    correctly_tagged_tokens += most_common_pos_count
    total_tokens += sum(pos_dict.values())

# Calculate accuracy
most_common_tag_accuracy = correctly_tagged_tokens / total_tokens * 100

#naive tagging on the test data
# Function to apply the naive tagging approach to the test data and compute accuracy
def evaluate_naive_tagger(test_file, term_postag_count):
    correctly_tagged_tokens = 0
    total_tokens = 0

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            word_pos_pairs = line.strip().split()
            for pair in word_pos_pairs:
                token, actual_pos_tag = pair.split('/', 1)  # Separate token and actual POS tag
                total_tokens += 1
                
                # Apply the naive tagging approach
                if token in term_postag_count:
                    # Get the most common tag for this token based on training data
                    most_common_pos = max(term_postag_count[token], key=term_postag_count[token].get)
                else:
                    # Enhanced handling for unknown words
                    if token.isdigit():
                        most_common_pos = 'CD'  # Cardinal number
                    elif token[0].isupper():
                        most_common_pos = 'NNP'  # Proper noun
                    elif token in {".", ",", "!", "?"}:
                        most_common_pos = '.'  # Punctuation
                    elif token.endswith('ing'):
                        most_common_pos = 'VBG'  # Gerund or present participle verb
                    elif token.endswith('ed'):
                        most_common_pos = 'VBD'  # Past-tense verb
                    elif token.endswith('ly'):
                        most_common_pos = 'RB'   # Adverb
                    elif token.endswith('s'):
                        most_common_pos = 'NNS'  # Plural noun
                    elif token.endswith('ion'):
                        most_common_pos = 'NN'   # Noun
                    else:
                        # Default to the most common POS tag overall
                        most_common_pos = most_common_tag
                
                # Compare with the gold standard (actual) POS tag
                if most_common_pos == actual_pos_tag:
                    correctly_tagged_tokens += 1

    # Calculate and return accuracy
    accuracy = correctly_tagged_tokens / total_tokens * 100
    print(f"Accuracy of Naive Tagging Approach on Test Data: {accuracy:.2f}%")
    return accuracy

# Check if test data file is provided and run evaluation
if '-t' in opts:
    test_file = opts['-t']
    evaluate_naive_tagger(test_file, term_postag_count)
else:
    print("\n** WARNING: No test data file specified (opt: -t FILE) **", file=sys.stderr)

# Print results
print("\nMost Common POS Tag:", most_common_tag)
print("Accuracy of Single-Tag Method (using only the most frequent tag overall):", f"{accuracy_score:.2%}")
print("Accuracy of Most-Common-Tag-Per-Word Method:", f"{most_common_tag_accuracy:.2f}%")
evaluate_naive_tagger("test_data.txt", term_postag_count)