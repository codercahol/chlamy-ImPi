# Plans for Chlamydomonas ImPi

The project goal is to analyze the large amount of image data taken by Carolyne and turn it into useful insights.

There are three important pieces to this:

1. Run the analysis over the large amount of image data
2. Make the analysis results useful and clear
3. Get rigorous and interesting results from the analysis
4. validate that the experiments being run presently are legit
5. something on the camera that validates the data -- 

In order to accomplish these goals I will

1. develop a script that can access the cloud data, save intermediate results, and prototype the analysis on subsets of the total dataset. Computing the analysis over the whole workload isn't very time-sensitive so it's ok if it takes a day or two to run.

2. develop an online UI with streamlit and altair interactive plots that is well-labelled and lets the user explore the dataset as desired.

3. I will need to validate the computations are matching expected behavior for known strains and staying within physical/biological bounds of the inferred values. Additionally, I will explore different time-series analyses and dimensionality reduction techniques to find over-arching patterns in the large dataset.

**
4. what is the base rate of plating error (how many spots per plate should be expected?) (maybe from that some threshold to re-try plating). what is the variability of the measurements for the same strain (for a given set of fixed conditions.) 

**
5. fv/fm of wild types (WTF & CC4533; they are identical) > 0.6; YII > 0.05 at end of experiment in the light, and low enough misplating error, some cmdline script
   - fv/fm for just the first point, at t0
   - pt-wise std dev at the end of the expt should be 
   - grid of blanks -> where to put
   - why does my threshold vs the nb thresholding mask look differnet
   - install python on the camera comp?
   - DoE
   - acceptable plating rate?
   - acceptable variability of yield for WT's?
     - std dev after some time
   - sanity check of WT's within some range (just check CC4533)

## I. 

## II.

## III.

next step here is to really rigorously validate that the cropping and data gathering is robust across all the different files. This requires moving forward together with (I). An important immediate milestone is to compute the estimated failure rate and propose a statistical test for when an experiment is pooched bc the robot didn't properly pcik + re-apply the colonies


there will be a new 384 well plate w/different spacing of colonies that will need to be re-calibrated.

let Adrien know if there are outliers for the WTs.

GSD -- and let Adrien know if I can't ASAP
meeting next Tuesday

9/10/23
First step is to get the estimated misplating for a single plate (and number of failed images/when they were) and also add the sanity checks for inferred values
- re-size altair graphs
- check other trials
- label strain with a new column 'replicate'
- a) estimate misplating
- b) use graphs to do sanity checks on data/familiarize

Next is to apply the modularized functions to some limited number of randomly selected trials and see how it does 
- set a custom image storage path for each test image and run it in slurm

(by manually inspecting the output intermediates images) --> utilities needed to:
- name the files and their derivatives appropriately
- get the list of all the relevant cloud files
- download the intermediate outputs nicely from the server
- run the processing on the server, in parallel, and save it to the appropriate locations