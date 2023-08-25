---
title: "Pre-requisites for Intermediate Machine Learning course"
author: "Andrew Parnell"
output: html_document
---

In preparation for the course please install the following, preferably in the below suggested order. Make sure you run these as soon as possible to avoid falling behind. If you have problems don't worry - we can fix them during the afternoon session on the first day

Remember you will need your own personal computer with administrator access for the duration of the course and a good internet connection.

As this module will be delivered online please install [Zoom](https://www.zoom.us) and [Slack](https://slack.com) to access the videos and interactive components of the course. All the Zoom links to the meeting will be posted to the Slack `#General` channel.

### Step 1

Install the following using the corresponding links

-	R: [http://www.r-project.org](http://www.r-project.org)

-	Rstudio (optional but recommended): [https://www.rstudio.com](https://www.rstudio.com)


### Step 2

The main package we will be using is `keras`. You can install it with

```{r,eval=FALSE}
install.packages('keras')
library(keras)
install_keras()
```

There is a nice introduction for installing `keras` and checking it works at https://cran.r-project.org/web/packages/keras/vignettes/. In particular copying the code and running the basic model there will help diagnose any problems with your installation

Now install the other packages we need:

```{r,eval=FALSE}
install.packages(c('nnet','ggplot2','BART','BASS','BayesGPfit','mclust','kernlab','pdfCluster','umap','fpc','tfdatasets','text2vec'))
```

### Troubleshooting

If you run into any problems please drop me a line at <andrew.parnell@gmail.com>.

