# Convolutional Neural Network for energy disaggregation of the Non-Intrusive Load Monitoring REDD Data Set

# The Redd Data Set

In the context of this project we make use of the REDD data set \cite{REDD}, a public data set for energy disaggregation research. The REDD dataset is not readily accessible since access can only be granted by the authors of \cite{REDD} at MIT. While REDD, is what made this project possible, it cannot be directly fed into the disaggregation algorithms developed/borrowed in this project. A data Pipeline is necessary to preprocess and feed the data for both training as well as testing purposes. While the training and disaggregation steps are significantly different in all of the methods that we use, the remaining steps are the same and are outlined in Figure~\ref{fig:redddata}

![](../poster/Figures/Qualitative.pdf)

\begin{figure}
\begin{center}
\includegraphics[width=30 pc]{../poster/Figures/Qualitative}
\caption{\footnotesize  \textbf{Figure}: The energy consumption of a household (Building 2 in the REDD dataset) during a period of 1 month} 

\label{fig:qualitative}
\end{center}
\end{figure}

# The redd data set consist of aggregated power from 6 houses with various sampling rates as well as  the power of a set of appliances per house. The recording is unfortunately not continuous in time and does not span the same time period for the all the houses. We downsample the data in order to align the appliances and the main meters time series. The preprocessing is specific to each algorithm used for prediction and is detailed later in this report. However the work flow common to each of the algorithm is shown in Figure~\

![](../poster/Figures/NILM_Data_Pipeline.pdf)

\begin{figure}
\begin{center}
\includegraphics[width=0.95\textwidth]{NILM_Data_Pipeline}
\caption*{\footnotesize  \textbf{Figure}: The NILM pipeline At each stage of the pipeline, results and data can be stored to or loaded from disk} \vspace*{-1 cm}
\end{center}
\end{figure}

# Convolutional Neural Network (ConvNet)

The implementation of the method presented in this section can be found in the notebook https://github.com/tperol/am207-NILM-project/blob/master/Report_convnet.ipynb. However the main codes are available in a separate repository (https://github.com/tperol/neuralnilm) to keep this final repository clean. Most of the preprocessing code are borrowed from Jack Kelly repository that was forked (https://github.com/JackKelly/neuralnilm). However the implementation of the python generator for the data augmentation on CPU, the ConvNet implementation (trained on GPU) and post processing for the metrics are our own implementation.

## 1- ConvNet introduction

Convolutional Neural Networks are similar to ordinary Neural Networks (multi-layer perceptrons). Each neuron receive an input, perform a dot product with its weights and follow this with a non-linearity (here we only use ReLu). The whole network has a loss function that is here the Root Mean Square (RMS) error (details later). The network implements the 'rectangle method'. From the input sequence we invert for the start time, the end time and the average power of only one appliance (see Figure~\ref{convnet_architecture}).

\begin{figure}
\begin{center}
\includegraphics[width=0.75\textwidth]{convnet_architecture}
\caption{A schematic representation of the architecture of the convolutional neural network} \label{convnet_architecture}

\end{center}

\end{figure}


Convolutional neural networks have revolutionized computer vision. From an image the convolutional layer learns through its weights low level features. In the case of an image the features detectors (filters) would be: horizontal lines, blobs etc. These filters are built using a small receptive field and share weights across the entire input, which makes them translation invariant. Similarly, in the case of time series, the filters extract low level feature in the time series. By experimenting we found that only using 16 of these filters gives a good predictive power to the ConvNet. This convolutional layer is then flatten and we use 2 hidden layers of 1024 and 512 neurons with ReLu activation function before the output layer of 3 neurons (start time, end time and average power).



## 2- Data pipeline

### 2.1 - Selecting appliances

We train each neural network per appliance. This is different from the CO and FHMM methods. For this report we only try to invert for the activation of the fridge and the microwave in the aggregated data. This two appliances have very different activation signatures (see Figure !!!).

### 2..2 - Selecting time sequences

We downsampled  the main meters and the submeters to 6 samples per seconds in order to have the aggregated and the submeter sequences properly aligned. We throw away any activation shorter than some threshold duration to avoid spurious spikes. For each sequence we use 512 samples (about 85 seconds of recording).

### 2.3 - Selecting houses

We choose to train the algorithm on house 1,2,3 and 6 and test the data on house 5.

### 2.4 - Dealing with unbalanced data: selecting aggregated data windows

We first extract using NILMTK libraries (http://nilmtk.github.io) the target appliance (fridge or microwave) activations in the time series. We concatenate the times series from house 1,2,3, and 6 for the training set and will test on house 5. We feed to our neural network algorithm (detailed later) balanced mini-batches of data sequences of aggregated data in which the fridge is activated and sequences in which it is not activated. This is a way to deal with unbalanced data -- there are more sequences where the fridge is not activated than sequences with the fridge activated. Most of the data pipeline used is borrowed from https://github.com/JackKelly/neuralnilm.

### 2.5 - Synthetic aggregated data

We use the method from Jack Kelly to create synthetic data. To create a single sequence of synthetic data, we start with two vectors of zeros: one vector will become the input to the net; the other will become the target. The length of each vector defines the ‘window width’ of data that the network sees. We go through five appliance classes and decide whether or not to add an activation of that class to the training sequence. There is a 50% chance that the target appliance will appear in the sequence and a 25% chance for each other ‘distractor’ appliance. For each selected appliance class, we randomly select an appliance activation and then randomly pick where to add that activation on the input vector. Distractor appliances can appear anywhere in the sequence (even if this means that only part of the activation will be included in the sequence). The target appliance activation must be completely contained within the sequence (unless it is too large to fit).
We ran neural networks with and without synthetic aggregated data. We found that synthetic data acts as a regulizer, it improves the scores on useen house.

We ran neural networks with and without synthetic aggregated data. We found that synthetic data acts as a regulizer, it improves the scores on useen house.

## 3 - Standardisation of the input data (aggregated data)

A typical step in the data pipeline of neural network is the standardization of data. For each sequences of 512 samples (= 85 seconds) we substract the mean to center the sequence. Furthermore every input sequence is divided by the standard deviation of a random sample in the training set. In this case we cannot divide each sequence by its own standard deviation because it would delete information about the scale of the signal. An example of 4 input sequences is shown in Figure~\ref{convnet_architecture}. 



## 4 - Output data (start time, end time and average power)

The output of the neural network is 3 neurons: start time, end time and average power. We rescale the time to the interval [0,1]. Therefore if the fridge starts in the middle of the input sequences the output of the first neuron is 0.5. If its stops after the end of the input window the ouput of the second neuron is set to 1. The third neuron is the average power during the activation period. Of course this is set to 0 when it is not activated during the input sequence. We also post process the data by setting any start time lower than 0 to 0 and end time higher than 1 to 1. We create a average power threshold set to 0.1 that indicates if the appliance was active or not (under the threshold the appliance is considered off, above it is considered on).
Here we show as an example the input data and the ouput calculated by a trained network. We compare this with the real appliance activation.



