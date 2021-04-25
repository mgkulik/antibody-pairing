# Ekm2

Our team is from Germany and we are from Albania, Brazil, Greece and Germany respectively.
We are PhD students in the CBDM group of the Instutute of Biology in the University Mainz and of course we also think that proteins are cool 🙌

We tried to circumvent the need for alignment and one-hot encoding. 
To do so, we used a packages called SGT - Sequence Graph Transform, to embed the protein sequences of the chains. 

The embedding gives a value for every pair of fĺetters in the amino acid alphabet of 20 amino acids. 
It returns 400 values, that depend on neighboring amino acids in the sequence. The embedddings are given as tuples of amino acids (A, F).

We fed this data to a grid search for a random forst and a fully connected NN. 

### Project Description
We are working on the `antibody-pairing` challenge.
To predict using our model, press "Open Application" on the left. 

### Scoreboard
You can track the performance of our predictor in the [challenge scoreboard](https://biolib.com/biohackathon/antibody-pairing-scoreboard/).
