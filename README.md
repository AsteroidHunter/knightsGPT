# The Knight's GPT
_Solving the Knight's Tour puzzle using an Autoregressive Transformer._

<img src="https://img.shields.io/badge/Status-Complete-orange" alt="Status" height="40">
<img src="https://img.shields.io/badge/Currently_Working_On-Training_more_models-8A2BE2" alt="Currently Working On" height="30">

## tl;dr 
- The [Knight's Tour](https://en.wikipedia.org/wiki/Knight%27s_tour) puzzle can be summarized concisely using the following question: _Given the rules of chess, is it possible for a knight to traverse the board and visit each square only once?_
- Inspired by [Li et al. (2022)](https://arxiv.org/pdf/2210.13382) and [Ruoss et al. (2022)](https://arxiv.org/pdf/2402.04494v1), I decided to test if an autoregressive transformer model trained on millions of Knight's tour, **encoded as merely linear indices representing positions on the chessboard**, could successfully learn how to solve the puzzle.
- I used three algorithms — Warnsdorff, backtracking with Warnsdorff, and recursive backtracking — to generate little over 17 million unique Knight's tours.
- I augmented the dataset by finding all symmetric tours for and expanded the total number of tours to 136 million. Of these, ~100 million were isolated for training.
- I configured a ~57 million parameter GPT-2 model and trained it on 1%, 10%, and 25% of the data for an epoch on two P100 GPUs
- To test model generalization, I used [Parberry's algorithm](https://www.sciencedirect.com/science/article/pii/S0166218X96000108) to generate 1191 partial tours (these partial tour patterns are not present in the training dataset).
- **Preliminary Results**
    - The models virtually never make illegal Knight moves;  (error 
    - All the models are able to solve tours from unseen starting positions (even starting positions omitted from the training data)
    - The models show some level of generalization as they are able to solve 815-999 out of the 1191 Parberry tours.
    - There is weak evidence of scaling leading to better model performance as the model version trained on 25% of the training data is able to solve the largest number of unseen Parberry tours.
 
## Further improvements
#### 1 Better tour generation
In the beginning, we didn’t consider the board’s symmetry when generating tours. This meant that a lot of tours generated using the Warnsdorff and Warnsdorff & backtracking algorithm ended up being identical (due to symmetry). Later, we primarily used the backtracking-only algorithm to generate unique tours, and we would have saved a lot of compute and time had we started with this algorithm earlier.

A more fundamental issue is the structure of the tours due to the backtracking algorithm (the first ∼45 steps of tours beginning from a particular position tend to be identical, see figure 10). This makes the model default to a particular solution instead of creatively exploring different possible solutions (since it sees those extremely common sequences several times). The solution here would be exploring other algorithms which generate more diverse tours; we haven’t found any other algorithm yet which can generate millions of unique solutions in a reasonable amount of time.
#### Masked modeling to make the model predict moves in the middle of the tour
Can the model complete tours if the first and last position are provided? Or, if the first n and last m positions are provided? Or, can the model fill in a tour when the first n moves, m moves in the middle, and o moves in the end are given? During training, a special masking token could be added to test tour generation under increased constraints.
#### Training on varying sequence lengths to test if the model can learn tours of varying lengths on different sized chessboards
The most efficient and successful Knight’s tour solving algorithms are able to solve tours for any chessboard (for which feasible solutions exist). Our models weren’t able to do so as they weren’t trained on indices belonging to larger or smaller chessboards. Generating tours of varying lengths for various chessboards, padding them, and then training the model on such data would highlight if autoregressive models are able to generalize even further. For testing, boards not present in the training data could be used.
#### Better test of scaling
Training models of varying sizes and ones trained on varying fractions of the training set could be used to test the combination of parameter and dataset size that enables these models to successfully solve combinatorial problems such as the Knight’s tour. There may exist a threshold after which models are virtually always successful at solving all unseen Parberry tours.
#### Using interpretability methods to better understand the model
Linear probes or sparse autoencoders could be employed to gauge if the model has developed an internal representation of the board state or a graph structure. If such a board or graph structure is found, then it may suggest that autoregressive transformers are able to conceptualize spatial relations despite not having been explicitly trained on two-dimensional chessboard data. Such methods would also help highlight the precise reasons why the model fails at certain unseen tours.
