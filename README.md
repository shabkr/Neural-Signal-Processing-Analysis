# Neural Signal Processing & Analysis

Python Implementation of the Matlab Problem Sets from MX Cohen's Neural Signal Processing Course, titled *"Complete neural signal processing and analysis: Zero to hero"*

[The complete course by MX Cohen can be found here.](https://www.udemy.com/course/solved-challenges-ants/) Please, refer to the original course lectures to learn and follow along with these exercises.

Thanks to user [dxganta](https://github.com/dxganta) for creating the original Python Implementation of Sections 1 through 9. User [shabkr](https://github.com/shabkr) is expanding on these original implementations. I've tried to keep everything as similar as possible to the original Matlab scripts, but there may be places were some code is slightly different due to differences in Python and Matlab, or make use of the features of Jupyter notebooks (i.e. embedded Markdown).

## Section status

- [x] Section 1: Introduction
  - *not applicable, no Matlab exercises*
- [ ] Section 2: The basics of neural signal processing
- [ ] Section 3: Simulating time series signals and noise
- [ ] Section 4: Time-domain analyses
- [ ] Section 5: Static spectral analysis
- [ ] Section 6: More on static spectral analyses
- [ ] Section 7: Time-frequency analysis
- [ ] Section 8: More on time-frequency analysis
- [ ] Section 9: Synchronization analyses
- [ ] Section 10: More on synchronization analyses
- [ ] Section 11: Permutation-based statistics
- [ ] Section 12: More on permutation testing statistics
- [ ] Section 13: Multivariate components analysis
- [ ] Section 14: Bonus section

## Dependencies

The following Python libraries will be required in this course:

- jupyter (for running the jupyter notebooks that contain the exercises)

- matplotlib (for pyplot)
- mne
- numpy
- scipy (for loadmat from scipy.io)

## How to use

Instructions below should work on all operating systems, unless otherwise specified.

### 0. Join the Udemy course

[The complete course by MX Cohen can be found here](https://www.udemy.com/course/solved-challenges-ants/). The Udemy course has a cost, but I have seen it go on sale before. Alternatively, some local libraries and universities offer their members access to Udemy, so be sure to check out those options if you are a student or otherwise on a budget!

### 1. Download

Clone the repository, or download the zip! Launch your operating system's terminal and navigate to inside the folder before proceeding.

### 2. Create a Python Environment and install the requirements

We'll use python's `venv` tool to build our environment. You can find [thorough virtual environment instructions on the Python Website](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments), but there are also brief instructions below:

- For Unix/macOS systems

```bash
cd NEURAL SIGNAL PROCESSING COURSE DIRECTORY #if you aren't there already
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

- For Windows systems

```powershell
cd NEURAL SIGNAL PROCESSING COURSE DIRECTORY #if you aren't there already
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

### 3. Launch Jupyter and begin learning!

**IMPORTANT.** Make sure your environment is active with `source .venv/bin/activate` (Unix/macOS) or `.venv\Scripts\activate` (Windows) before you launch your course notebooks!

The command `python3 -m jupyter notebook` (Unix/macOS) or `py -m jupyter notebook` (Windows) will launch your Jupyter Notebook session, and you can navigate to the exercise you are on. Ideally have both Jupyter Notebook and the Udemy Course open side by side so that you can work alongside the relevant video. Happy learning!
