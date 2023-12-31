---
title: "Pre-requisites for Intermediate Machine Learning course"
author: "Andrew Parnell"
output: html_document
---

In preparation for the course please install the following, preferably in the below suggested order. Make sure you run these as soon as possible to avoid falling behind. If you have problems don't worry - we can fix them during the afternoon session on the first day

Remember you will need your own personal computer with administrator access for the duration of the course and a good internet connection.

As this module will be delivered online please install [Zoom](https://www.zoom.us) and [Slack](https://slack.com) to access the videos and interactive components of the course. All the Zoom links to the meeting will be posted to the Slack `#zoom-links` channel. 

### Step 1

Install the following using the corresponding links

-	R: [https://www.r-project.org](https://www.r-project.org)

-	Rstudio (optional but recommended): [https://posit.co/products/open-source/rstudio/](https://posit.co/products/open-source/rstudio/)


### Step 2

Install Python, tensorflow, and keras by following the instructions at: https://tensorflow.rstudio.com/install/

At the end you should be able to run

```
library(tensorflow)
library(keras)
```

without error.

### Step 3

Now install the other packages we need:

```{r, eval=FALSE}
install.packages(c('tidyverse','nnet','BART','BASS','BayesGPfit','mclust',
'kernlab','pdfCluster','umap','fpc','tfdatasets','text2vec', 'anomalize', 
'tsoutliers', 'stray', 'mlbench', 'tsibbledata'))
```

### Cloud version

If you are struggling to install keras or you have a computer that does not allow you administrator access you can try to install keras and the other packages in the cloud version of Rstudio at (https://posit.cloud). You need to create a (paid unfortunately) account and after that you can install keras with:

```{r,eval=FALSE}
install.packages('keras')
library(keras)
install_keras(method="virtualenv", envname="myenv", pip_options = "--no-cache-dir")
library(tensorflow)
install_tensorflow()
```

### Troubleshooting

If you run into any problems please drop me a line at <andrew.parnell@gmail.com>.

