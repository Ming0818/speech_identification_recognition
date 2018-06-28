import os
import numpy as np

dataDir = '/data/'

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    n = len(r)
    m = len(h)
    R = np.zeros((n+1, m+1))
    B = np.zeros((n+1, m+1))

    for i in range(n+1):
        for j in range(m+1):
            if i == 0:
                R[i][j] = j
            if j == 0:
                R[i][j] = i

    R[0][0] = 0

    for i in range(1,n+1):
        for j in range(1,m+1):
            dels = R[i-1,j]+1
            subs = R[i-1,j-1]
            if r[i-1] == h[j-1]:
                subs += 0
            else:
                subs += 1
            ins = R[i,j-1]+1

            dict = {'substitutions':subs, 'insertions':ins, 'deletions':dels}
            minimum = min(dict.items(), key=lambda x: x[1])
            R[i][j] = minimum[1]

            if minimum[0] == "deletions":
                B[i,j] = 1
            elif minimum[0] == "insertions":
                B[i,j] = 2
            else:
                B[i,j] = 3

    results = backtrack(B,R)

    if n == 0:
        results = {'substitutions':0, 'insertions':m, 'deletions':0}
    elif m == 0:
        results = {'substitutions':0, 'insertions':0, 'deletions':n}

    wer = 0
    if R[n][m] == 0 or n == 0:
        wer = 0
    else:
        wer = R[n][m]/n

    return wer, results['substitutions'], results['insertions'], results['deletions']

def backtrack(B, R):
    # key = {3:'up-left', 2:'left', 1:'up'}
    counts = {'substitutions':0, 'insertions':0, 'deletions':0}

    i = len(B)-1
    j = len(B[0])-1

    while i != 0 and j != 0:

        if B[i][j] == 3:
            if R[i][j] != R[i-1][j-1]:
                counts['substitutions'] += 1
            i = i - 1
            j = j - 1
        elif B[i][j] == 2:
            counts['insertions'] += 1
            i = i
            j = j - 1
        elif B[i][j] == 1:
            counts['deletions'] += 1
            i = i - 1
            j = j

    if i!=0 and j==0:
        counts['deletions'] += R[i][j]
    elif i==0 and j!=0:
        counts['insertions'] += R[i][j]

    return counts

def preproc(text):

    text = text.lower()
    parts = text.split()
    for idx in range(len(parts)):
        if '/' in parts[idx] and ':' in parts[idx]:
            parts[idx] = ""
        if '<' in parts[idx] and '>' in parts[idx]:
            parts[idx] = ""
        if '[' in parts[idx] and ']' in parts[idx]:
            parts[idx] = ""

    processed_text = ''
    for part in parts:
        if part != "":
            processed_text += part + " "

    for punc in ['!', '.', ',', ':', ';', '(', ')', '\'', '/', '"', '{', '}', '<', '>', '~', '[', ']', '-', '_', '?', '  ']:
        if punc == '  ':
            processed_text = processed_text.replace(punc, ' ')
        else:
            processed_text = processed_text.replace(punc, '')

    return processed_text


if __name__ == "__main__":

    wer_vals = []
    sum_wer = 0
    wer_count = 0

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:

            transcripts = open(os.path.join( dataDir, speaker, "transcripts.txt"), "r").read()
            Kaldi = open(os.path.join( dataDir, speaker, "transcripts.Google.txt"), "r").read()
            Google = open(os.path.join( dataDir, speaker, "transcripts.Kaldi.txt"), "r").read()

            trans_list = []
            transcript_lines = transcripts.split("\n")
            for transcript in transcript_lines:
                preprocessed = preproc(transcript)
                trans_list.append(preprocessed)

            kaldi_list = []
            kaldi_lines = Kaldi.split("\n")
            for kaldi_line in kaldi_lines:
                preprocessed = preproc(kaldi_line)
                kaldi_list.append(preprocessed)

            google_list = []
            google_lines = Google.split("\n")
            for google_line in google_lines:
                preprocessed = preproc(google_line)
                google_list.append(preprocessed)

            if len(trans_list) == 1:
                continue

            if len(kaldi_list) > 1:
                for i in range(len(kaldi_list)):
                    wer, subs, ins, dels = Levenshtein(trans_list[i].split(), kaldi_list[i].split())
                    print(speaker, "Kaldi", i, wer, 'S: {}'.format(subs), 'I: {}'.format(ins), 'D: {}'.format(dels))
                    wer_vals.append(wer)
                    sum_wer += wer
                    wer_count += 1

            if len(google_list) > 1:
                for i in range(len(google_list)):
                    wer, subs, ins, dels = Levenshtein(trans_list[i].split(), google_list[i].split())
                    print(speaker, "Google", i, wer, 'S: {}'.format(subs), 'I: {}'.format(ins), 'D: {}'.format(dels))
                    wer_vals.append(wer)
                    sum_wer += wer
                    wer_count += 1

    average = sum_wer/wer_count
    diff = 0
    for i in wer_vals:
        diff += ((i-average) ** 2)
    sd = (diff/wer_count) ** 0.5

    print('Average = {}'.format(average), 'Standard Deviation = {}'.format(sd))
