# Dropout-
In this experiment, we implement Dropout, a simple yet highly effective Regularization technique. We examine and implement the technique of Inverted Dropout (applied to neurons only while training unlike Vanilla Dropout where the criterion is evaluated for each neuron while testing) 

Dropout is an extremely effective, simple and recently introduced regularization technique by Srivastava et al. in Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014, JMLR) that complements the other methods (L1, L2, maxnorm). At each training iteration a dropout layer randomly removes some nodes in the network long with all of their incoming and outgoing connections. Dropout can be applied to hidden or input layer. 

The original paper which introduced the concept was published in 2012 by Hinton et al, [1207.0580] Improving neural networks by preventing co-adaptation of feature detectors, 2012 (probably the original paper on dropout)
 Dropout will randomly mute some neurons in the neural network and we therefore have sparse network which hugely decreases the possibility of overfitting. More importantly, the dropout will make the weights spread over the input features instead of focusing on some features.The possibility of muting neurons is often set as 0.5. When the dropout is 1.0, then our network simply shall not drop out any neurons. 


Randomly cutting units from the network is one way to test how robust it is. It does this by making it more difficult for units in a network to accidentally compensate for one another. Any given unit in a neural network may make an error, and there is a chance that the error could be nullified by another unit making an error which pushes back in the wrong direction. This would cause a situation where the network has the correct answer but for the wrong reasons, or arrived at the answer by the wrong process. Dropping units helps guard against this, ensuring that any correct answers the network arrives at are truly because the algorithm being used is successful, not the result of chance. 
