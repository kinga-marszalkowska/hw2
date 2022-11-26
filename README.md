# Supervised Machine Learning Challenge

## Goal

The objective of this assignment is to train a classifier, attempting to get good generalization performance. You can use whatever tools and methods you wish (within reason), but you're expected to do the work yourself.

## Data

Supervised learning, 10-dimensional input, binary (0/1) output.

These are the files provided to you.

* `train-io.txt.gz` is a dataset of 100000 training samples, one/line, with the first 10 numbers constituting the input and the last number the target output. FYI, the samples are iid.

* `test-i.txt.gz` is 10000 test cases, one/line, with the ten numbers on each line constituting an input.

* `test-o.txt.asc` is the 10000 correct labels, one/line, each line consisting of a single character, either `0` or `1`.  This file was encrypted using `gpg --symmetric --armor`; I will reveal the key after the assignment is over.

## Turn In

Please turn in (via Moodle) the following:

### Your labelling of the test set

* `test-o-hat.txt`, a file consisting of your best guess for the 10000 labels for the test cases, one/line, each line consisting of a single character, either `0` or `1`.

  **NOTE** This file should contain *exactly* 20000 characters, since each line is a single digit followed by a newline character. (Or on MS Windows, with its strange CR-LF two-character end of line business, 30000 characters, in which case I'll `dos2unix` it automatically.) Please be super fanatic about making sure this file is in the correct format. No extra comment line at the top, no extra blank lines at the end, no leading or trailing whitespace.

  Here's how you can check.
  ```
  $ file test-o-hat.txt
  test-o-hat.txt: ASCII text

  $ wc -l test-o-hat.txt
  10000

  $ wc -c test-o-hat.txt
  20000

  $ sort < test-o-hat.txt | uniq
  0
  1
  ```

* If it is not in the right format, my automated tools won't handle it.
* If my automated tools won't handle it, I'll have to deal with it manually.
* I don't want to deal with it manually.
* You don't want me to have to deal with it manually.
* Don't make me deal with it manually.

### Your Code, which I can read and run

* Source code to allow me to train a model and generate `labels.txt`, in a `.tgz` or `.zip` file. This should consist of *just* the source code you wrote, along with instructions for running it, including a list of what software needs to be installed first (e.g., R and various packages, or Octave and various packages, or pytorch, or whatever). Ideally the instructions would be to type `make labels.txt`, but if they’re more complicated that’s okay. The code can assume that `train.txt` and `test.txt` are present (uncompressed).

###  A Report

A report explaining what you did. This should include your guess (with justification) as to your error rate on the test set. Remember: *brevity is the soul of wit!* This should be in PDF format.  (If you used latex feel free to include the .tex source, and if you used Libreoffice or MS Word you can include its editable format. But do include the PDF too because that's easier for me to read and make notes on, and environmental differences can sometimes make it difficult for me to read your .docx or whatever file.)

* A document for me to read.
* Describing what you did (keeping in mind that brevity is the soul of wit)
* Graphs and tables that you generated while optimizing things is nice.
* Can be handwritten/scanned, e.g., the drawings or some graphs, if you want. *It's not a journal paper, so relax!* As long as I can read it. You can be very brief, please don't pad it out.

### Smarty Pants (optional)

For **extra credit**, a file `smarty-pants.txt` explaining what the hidden structure in the data is. Hint: you will not be able to just guess: if you find it you'll have done some exploratory data analysis.

### **DO NOT** include the data files I'm distributing

Please do *not* include the data files I'm distributing.  Or trivially transformed versions thereof; instead, include the transformation in your scripts for running the code as described above

### **DO NOT** include cruft

Also please do not include "generated" files, like compiled executables, .pyc files, .o files, intermediate data files, etc.

## FAQ

**Q:** Do we need to submit only a single classifier or we could submit multiple classifiers that are implemented using different algorithms?

**A:** You can submit multiple classifiers but you need to tell me which one is “it”, the one you think will generalize best, and put its predictions in `labels.txt`, with instructions for regenerating that particular file.  And you should describe what you did in your report.

**Q:** Do we need to code everything from scratch or it is fine if we use the machine learning libraries in python?

**A:** You can use ML libraries and such; you are not required to implement algorithms yourself.
If this makes your code very short, that is fine.

**Q:** Can you push back the deadline?

**A:** Yes, I can. But please don't spend too much time on this. And I need to pipeline my grading, so I do need to have a bunch on time.

**Q:** Doing full k-way k-fold cross validation is really slow, what should I do so my experiments are fast enough to make some progress?

**A:** Ideas: just do 1-way cross validation, thus making a noisier but much faster estimate. You can also try using a subset of the training set for speed, depending on the algorithm you’re using.

**Q:** Is it okay to use a lot of CPU time?

**A:** Sure, as long as *I’m* not paying your power bill. Be sure to keep track of what you did (like scientists do, with a lab notebook) so you can explain it in the report to impress me.

**Q:** Can I use a cluster, or the cloud?

**A:** Sure.  Note: this is just a supposed-to-be-fun assignment, not a dataset for curing cancer.  So if you’re having fun, and it’s a learning experience, great!  But you don’t *need* to go overboard.

> Those who do not want to imitate anything, produce nothing.  
>                 —Salvador Dali

> Measuring programming progress by lines of code is like measuring aircraft building progress by weight.  
>                 —Bill Gates
