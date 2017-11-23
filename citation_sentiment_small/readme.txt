For each instance, you'll find that there are four sentences, the citing sentence to be labelled itself, it's previous sentence and the next two sentences. Each of the context sentence is labelled with whether it was useful in making the sentiment decision or not. The paper ids correspond to AAN Ids, that you should be able to get here: http://clair.eecs.umich.edu/aan/index.php. Given this, here is the format of the file:

<citing_id>  <cited_id>  <year> <sentences with labels> <purpose_label> <polarity_label>

The purpose label can be a digit from 1 - 6, which correspond to the following labels:

1 - Criticizing
2 - Comparison
3 - Use
4 - Substantiating
5 - Basis
6 - Neutral

The polarity label can be a digit from 1 - 3 which corresponds to the following labels:

1 - neutral
2 - positive
3 - negative

Please mail any questions to: rahuljha@umich.edu
