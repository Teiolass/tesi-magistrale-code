#import "template.typ": *

#show: report.with(
  title: "Thesis",
  authors: (
    (
      name: "Alessio Marchetti",
      organization: [Università di Pisa],
      email: "a.marchetti32@studenti.unipi.it",
    ),
  ),
  bibliography-file: "refs.bib",
)


#let ds = math.cal("D")
#let R  = math.bb("R")

= Introduction

= Related

== Machine Learning for Medical Data

Qui si parla di drAI

== LLM and Transformers

The developement of predictor models for categorical sequential data has historically been deeply
intertwined with the text-generation and machine-translation tasks. In the most commons approaches,
the text is divided in smaller portions called _tokens_, which usually represent a word, a suffix, a
punctuation symbol, or in general an indivisible part of the language. Typical values for the number
of tokens in a single language are in the order of few tens of thousands.

The text generation task is the problem of generating a text answer to a text prompt, which can be a
question or some kind instruction. The machine translation task is the problem of generating the
translation to some given input text. These two tasks are very similar, and for clarity we will
focus on the translation one  in the following discussion.

Handling sequence of data of arbitrary length has always been a challenge in the context of deep
learning, due to the fixed size of the input and output layers in Neural Networks. The first
effective solution has been the idea of recurrent architectures.

The task is split in two parts: in the first part the model gets in input the tokens of the prompt
to translate. In the second part the answer is generated in an autoregressive manner: starting from
a special `<start>` token, the model generates the next token. The new token is then concatenated to
the previous ones, and is given to the model, which generates a next token. This process continues
until a special `<end>` token is generated, or the generated sequence length exceeds a fixed limit.

=== Recurrent Models and Attention

A recurrent model is built in such a way that it can take in input a hidden memory vector $h$, and a
token $t$, and then outputs $h'$, an updated version of $h$. The new hidden vector is then fed to
the model together with the next token.

The process is started with a fixed initial vector $h_0$, and let's assume the prompt is made by the
token sequence $t_0, dots, t_n$. Then given the recurrent model $f$, the first hidden vector
generated is $h_1 = f(h_0, t_0)$, and in general we define $h_(i+i) = f(h_i, t_i)$ for $i=1...n$.
This tackles the first part of the problem. 

A very simple form for the model $f$ is the following. Let $e_1, ..., e_N$ be a sequence of fixed
$k$-dimensional vectors (these are called _embeddings_), where $N$ is the number of possible tokens
and $k$ is a fixed integer. Let $sigma$ be a non linear activation function, typical choices are the
Rectified Linear Unit (ReLU) or the sigmoid function. Let $W_h$ and $W_t$ be two $n times n$
matrices a $b$ a $k$-dimensional vector. Then if $h$ is a $k$-dimensional hidden state, and $t$ a
token corresponding to the vector $e_j$, we set $f(h,t) = sigma(W_h h + W_t e_j + b)$.

Other more complex architectures has been explored, but this simple presented one is the base for
all of them.

For the second part we start with a the `<start>` token which we will call $g_0$. From this we
generate a new hidden state $hat(h)_1 = f(h_(n+1), g_0)$. With a classical NN model $f_"gen"$ we can
use $hat(h)_1$ to generate a probability distribution $f_"gen" (hat(h)_1)$ over all the possible
tokens. We can then extract a new token from this distribution. Usual approaches consists in taking
the maximum probability token, a plain sampling over the distribution, or some hybrid methods in
which the sample is performed only on the most proabable tokens. The result of this process is a new
token $g_1$. From this point onward we repeat the process, generating $hat(h)_2 = f(hat(h)_1, g_1)$
and from the new hidden state a new token, and so on.

The training of this kind of models is performed noticing that it defines a probability distribution
over all the possible answer sentences. In fact each new token is generated from a probability
distribution given the previous ones. This is effectively a decomposition of the probability of the
whole sentence. Thus the training is performed maximizing the likelihood of a dataset of human
generated pairs of input and outputs.

As in most of trainings of Neural Networks, the optimization is done through Stochastic Gradient
Descent and backpropagation. This reveals the biggest weakness of recurrent architectures. Assuming
the model has parsed $n$ tokens between inputs and outputs, which means that reasonable values for
$n$ could range from a few dozens to few hundreds or even more, the gradient for the first
application of $f$ has to go through all the $n$ layers that follows to arrive to the loss. During
all these steps the quality of the gradient is degraded, usually leading to the phenomenons of
exploding gradients or vanishing gradients. The idea is that the flow of the gradient is a dynamical
system which can be mostly described by a multiplication by a a matrix. If these matrices
consistently increase the magnitude of the gradient, the gradient will become very large, and thus a
step of SGD will move the model parameters in a region in which the function cannot be approximated
by the gradient in the starting point. This means that the update is meaningless. If instead the
matrices consistently decrease the magnitude of the gradient, the update of the SGD will be near
zero, and the model will fail to converge.

Several attempts to solve this problem have been tried by modifing the inner architecture of the
recurrent net, and the most notable examples of that are the Gate Recurrent Units (GRUs) and the
Long Short Term Memory models (LSTMs). These are however only mitigations of the problem, and not a
complete solution.

One first possible modification, that does not solve the gradient problem but still improves the
performances of the model is using two different models for the two parts of the task. These two
models have the same structure of the previous recurrent ones, but they are used for two different
objectives. The first part of the problem is to encode the input tokens into a single hidden vector,
and thus the model is called encoder. The second part of the problem is to take the hidden vector
and generate autoregressively the output, in fact decoding the vector into tokens. This second model
is thus called decoder.

A very import step for the evolution of these architectures has been in noticing that the hidden
state during the decoder phase of the process has two distinct functions: remembering the input, and
remembering at which point of the output generation the model is at. This suggest that the two
functions could be carried out by two different vectors. This can be easily accomplished by having
the decoder model be a function of a decoder hidden state, a context vector, which is the final
output of the encoder, and the previous token. The result is that the decoder is unable to forget
the encoder output, because it is not modified during the recurrence.

The next step has been improving this mechanism. In the translation task, it's very common for a
word in the output to map directly to a single word in the input, and more generally small groups of
words in the outputs correspond to small groups of words in the input. On the other hand most of the
data produced by the encoder, the hidden states, is discarded, keeping only the last hidden state.
As a consequence an arbitrary large amount of data, the input tokens, has to be compressed in a
single hidden state, whose size is fixed by the model architecture. The following idea improves over
this direction.

Let's assume that the encoder has generated the hidden states $h_0, ..., h_n$, and let $hat(h)$ be
the current hidden state in the decoder. In the previous architecture we would need a fixed context
vector generated from the encoder. What we do now is to generate a context vector dependent on
$hat(h)$. The first step is to have a measure of how important is a given hidden state $h_i$
according to $hat(h)$. In the most simple version of the architecture, this measure is simply the
dot product $h_i^top hat(h)$, but other differentiable functions of the two vectors could be used.
Once the importances $x_0, ..., x_n$ are given, they are converted on a distribution probability
over the hidden vectors of the encoder. This is done through the SoftMax function:
$ alpha_i = (exp (x_i)) / (sum_j exp (x_j)). $ 
The idea is that the probability distribution measures how relevant is each hidden state for the
current prediction, and thus how much attention should be given to each of them. The distribution is
in fact called attention. With this in mind, the context vector is simply the mean of the hiddden
states, weighted through the attention:
$ c = sum_i alpha_i h_i. $

This is very important because attention allows the gradient to "take shortcuts" to get to the first
layers without having to pass through all the following ones.

One importnt way to scale up in size and complextity all the previous models is to have several
layer the recurrent architectures. These leads to have a first layer similar to the ones described
before: it takes as an input the previous hidden state, the embedding of a token, and in some cases
a contex vector. Each of the subsequent layers takes in input the previous hidden state for that
layer, the hidden state of the previous layer, and sometimes the context vector.

=== Transformers

The next iteration on these design led to the current State of the Art models like GPT-3.5 or Llama.
The idea is to get rid entirely of the recurrent part, focusing only on the attention. The heart of
these models is the transformer. It is a layer that takes as an input a set of $d$-dimensional
vectors $e_1,..., e_n$ and returns a set of vectors of the same size. By means of a multiplication
by some fixed matrices $W_k$, $W_q$ and $W_v$ of size $d times d_h$ where $d_h$ is called hidden
size, the vectors $e_i$ are mapped to three sets of vectors: keys $K_i = W_k e_i$, queries $Q_i =
W_q e_i$ and values $V_i = W_v e_i$. The keys perform the same role of the hidden states $h_i$ when
choosing the importances in the previous discussion. The queries are the anologous of the vector
$hat(h)$, while the values are the vectors that will be combined through a weighted average.

For each query vector, a measure of importance is computed as before as the dot product with each
key value. This result is often scaled by the square root of the number of dimensions $d_h$, which
while it does not improve the expressiveness of the model, it balance the fact that increasing $d_h$
is reflected in bigger dot products. These importances are then processed in a SoftMax function to
give the attention probability distribution:
$ alpha_i = "SoftMax"((Q_i^top K_0, dots, Q_i^top K_n) / sqrt(d_h)) $ 
where 
$ "SoftMax"(y_1,...,y_n)_i = (exp(y_i)) / (sum_j exp (y_j)). $
The output is again obtained by means of a convex combination:
$ O_i = sum_j alpha_(i,j) V_j $

The previous relationships can be expressed in a more compact form seeing the vectors $K_i$, $Q_i$
and $V_i$ as column of the matrices $K$, $Q$ and $V$. Thus we obtain:
$ O = "SoftMax" ((Q K^top)/sqrt(d_h))V $ <self-attn>

Since the keys, queries and values are obtained from the same set of vectors $e_i$, the expression
@self-attn is called self-attention head.

Usually more than a single self-attention head is present in a transformer layer. In an
$m$-headed transformer there are $m$ triples of matrices $W_k^((i))$, $W_q^((i))$ and $W_v^((i))$,
and each of them is used to compute the respective output $O^((i))$ for $i=1,..., m$. These outputs
are concatenated and brought back to the right dimension by means of a linear projection through a
matrix $W_o$ of size $m d_h times d$:
$ y = "Concat"(O^((1)),..., O^((m))) W_o. $

Multiple transformer layers can be stacked on top of each other. Since the only source of
non-linearity is the SoftMax function, which appears only in the coefficients of the values, vectors
usually go through a Feed Forward Network (FFN) between a transformer layer and the other. These
networks are usually very simple, often in the form
$ "FFN"(y_i) = W_1 sigma(W_2 y_1 + b) $
where $sigma$ is an activation function, usually a ReLU, and $W_1, W_2, b$ are parameters of
suitable size. This component is also called Multi-Layer Perceptron (MLP).

As in most deep learning models, skip connections and layer normalizations are also added.
#note("Di più?")

In the recurrent models the order of the tokens was given to the model implicitely, because the
their embedding were fed in the right order. The same does not apply in transformer models: a
permutation of the inputs leads just to the same permutation on the outputs. In other words the
transformer cannot distinguish the order of the inputs by itself, and additional informations are
needed. While this can add to the complexity of the model, it also means Transformers are more
flexible by recurrent models.

The most common way to add temporal information is performed by adding to each input vector of the
transformer $x_i$ a fixed vector dependent by time only $p_i$. Thus while $x_i$ changes for each run
of the transformer, which will have different inputs, $p_i$ is fixed among all these runs. There are
different choices for $p_i$. One is to have it completely learnable by SGD. Another is to fix it,
usually to a vector in the following form:
$ p_i = mat(sin((omega i) / 2^0); cos((omega i) / 2^0); sin((omega i) / 2^1); cos((omega i) / 2^1);
  dots.v; sin((omega i) / 2^d); cos((omega i) / 2^d)) $
This kind of vectors has showed to perform well. A possible line of reasoning for its justification
is that vectors of sines and cosines can be translated in time with just linear transformations,
which are easily performed within the model.

We observe that the length of the path between each output vector and each input embedding is not
dependent by the distance between the corresponding token, but they interact directly in the
attention computation. This completely solves the biggest issue of the recurrent models.

Another big advantage of transformer based models respect to recurrent ones is that all the
computations can be carried out in parallel, instead of having to find the hidden state of the
previous iteration in order to be able to start the next one. This allows a better usage of GPUs and
therefore smaller training times.

Moreover, in a training phase, all the outputs can be computed in one single sweep. This should be
done carefully though, because the prediction of the $i$-th cannot be performed by having as input
the $i+1$-th token, or even the $i$-th token itself. This can be done by tweaking the attention
matrices, imposing that the attention for the $i$-th query and the $j$-th key are zero whenever $i
<= j$. The easiest way to accomplish this is by setting the corresponding importances to minus
infinity, which in turn will translate to zero through the SoftMax function. This correspond to add
to the importance matrices some upper triangular matrices.

Attention masks can be useful for another purpose as well. Until now we considered a single pair
input/output going through the model at each step of the training process. However usually deep
learning models are trained using minibatches of data for each step. This can be accomplished by
running multiple "single pass" steps, one for each element of the minibatch, and accumulating the
gradient. It may be more perfomant to deal with the minibatch in a single pass. This can be done by
concatenating all the data for the minibatch in a new dimension of the matrices, and the model
doesn't change much to accomodate this modification. However in general the sequence length will not
be constant through the minibatch. Thus all the sequences are padded to the max length within the
minibatch, and the attention mask is tweaked to avoid that any output depends on a padding token.

All of this tweaks allows the model to grow in size while still being trainable. In the last year in
fact there have been proposed several of such models, varying in size between a few billions
parameters to hundred of billions of parameters. These are called Large Language Models, or LLMs.
Examples are the GPT family developed by OpenAI, and the Llama family, developed by Meta.

=== The Llama-2 Models

The Llama-2 are a family of LLMs sharing the same underlying architecture and available in different
sizes: 7B, 13B, 34B and 70B, where the number counts the number of trainable parameters of the
model. The architecture is the same transformer-based decoder described in the previous section,
with the following modifications.
1. Normalization Layer... #note("Questo qui non c'è in Kelso")

2. The positional encoding have been replaced by Rotary Position Embeddings. The idea is to add
  positional information at each layer of the model, modifying the key and query vectors by a
  suitable function dependent by time. Let this modification be represented by the functions 
  $f_k(y_n, n)$ and $f_q(y_n, n)$, where $y_n$ is the $n$-th vector of the previous layeror the
  token embedding  mutliplied by the matrix $W_k$ or $W_q$. Let's assume that the hidden dimension
  to be an even number $d$, which is reasonable since for performance reasons typical values of
  dimensions are large powers of two or some of their multiples. Then a good choice could be
  $ f_({q, k})(y_n, n) = R_(Theta, n) y_m $
  where $R_(Theta, n)$ is a matrix in the following form:
  $ R_(Theta, n) = mat(
      R(n theta_1), 0, ..., 0;
      0, R(n theta_2), ..., 0;
      dots.v, dots.v, dots.down, dots.v;
      0,0, ..., R(n theta_(d slash 2));
    ),\ 
  R(theta) = mat(cos theta, -sin theta; sin theta, cos theta),
  $
  and $Theta = { theta_1, ..., theta_(d slash 2) }$ with $theta_i = 10000^(-2(i-1) slash d)$.

  This choice has a nice property. Let $x_n$ and $x_m$ be the outputs of the previous layer. Then
  the attention importance relative to the $n$-th query and $m$-th key using the Rotary Embedding
  can be written as
  $ I_(n,m) &= (R_(Theta, n)W_q x_n)^top (R_(Theta, m)W_k x_m) \ 
            &=  x_n^top W_q^top R_(Theta, m-n) W_k x_m.
  $ <rotary-importance>
  The matrix $R_(Theta, m-n)$ is an orthogonal matrix, which helps the flowing of the gradient
  during training. Moreover from equation @rotary-importance we can see that this is a kind of
  relative encoding, which, in contrast with the absolute encoding, add informations only about the
  distance of two tokens, and not their position in the whole sentence. For many tasks this is a
  desirable property. One final remark is that the matrices $R_(Theta, n)$ are very sparse, and thus
  the computation of the products can be performed in a $cal("O")(d)$ complexity.

3. The MLP layers has been replaced with a Swish Gated Linear Unit (SwiGLU). The function $"Swish"$
  is defined as 
  $ "Swish"(x) = x sigma(x) $
  where $sigma$ is the sigmoid function. Given a hidden dimension of the model $d_h$ and a new
  dimension $d_"MLP"$ for the MLP, we define the Gated Linear Unit as be parametrized by the
  $W_"gate"$, $W_"up"$ and $W_"down"$, of size $d_h times d_"MLP"$ for the first two, and $d_"MLP"
  times d_h$ for the last one. Let's denote the component-wise multiplication of two vectors with
  the symbol $dot.circle$. Then the layer is defined as
  $ "MLP"(x) = W_"down" ("Swish"(W_"gate" x) dot.circle W_"up" x) $
  This approach has been proven to have a better behaviour than a plain MLP layer.

  == Explainability 

- Carrellata di roba
- DrXAI

= Problem Description

== Data Type

Let $cal("C")$ be a finite set of discrete elements called _codes_. A visit is defined as any subset
of $cal("C")$. A _patient_ is defined as a finite ordered sequence of visits. A code is allowed to
appear more than once within different visits of the same patient. Let $cal("V")$ be the set of all
possible visits, and $cal("P")$ be the set of all possible patient.

A dataset #ds consists of a sequence of patients from $cal("P")$.

We assume that there exists a probability distribution $mu$ on the set $cal("P")$, which measures
the plausibility of each patient in a Bayesian sense, and we assume that #ds is a sequence of
independent patients $p_1, dots, p_N$ sampled from that distribution.

We can factor $mu$ over each visit in the following way: let $p=(v_1, dots, v_k)$ be a patient. Then
we decomposition is given by
$ mu(p) = mu_0(v_1) mu_"next" (v_2 | (v_1)) mu_"next" (v_3 | (v_1, v_2)) dots.c mu_"next" (v_k | (v_1,
  dots, v_(k-1))) $
where $mu_0$ is a probability distribution over the initial visits $cal("V")$, and $mu_"next"$ is a
probability distribution over the next visit given the preceeding ones. In other words $mu_"next" (v
| p')$ is the probability of $v$ being the next observed visit after seeing all the visits in $p'$.
#note("Probabilmente ci dovrebbe essere una 'visita terminale' per dire quando la sequnza finisce.")

A predictor model $h(dot.c, theta)$ parametrized by a vector of parameters $theta in #R^d$, for some
number of dimensions $d$, is a function
$ h(dot.c, theta) :: cal("P") --> [0,1]^(|cal("C")|). $
The underlying idea is for $h$ to be a predictor of the next visit for a given patient. In fact we
can see $h(p, theta)$ as a distribution on the visits $cal("V")$, where we assume that each code is
chosen independently from each other, and the $i$-th code has a probability of appearing given by
the $i$-th component of $h(p, theta)$. For example the vector $x=(x_1, dots, x_(|cal("C")|))$ assigns
to the visit $v=(c_(i_1), dots, c_(i_n))$ the probability $product _j x_(i_j)$. In the following
discussion we will identify the vector $x$ with its distribution on $cal("V")$.

The objective of a training process is to find an optimal parameter $hat(theta)$ for which the
function $h(dot.c, hat(theta))$ is the best approximation of the distribution $mu_"next"$. Since the
latter is never known, we will use the empirical distribution found in the dataset #ds as its proxy.

== Ontology

#let codes_in = $cal("C")_"in"$
#let codes = $cal("C")$
#let tree = $cal("T")$ 
#let patients = $cal("P")$
#let visits = $cal("v")$

In many fields, and in particular the medical one, which is of our interest, some categorical data
is not just made of indistinguishable labels, but rather by leaf-nodes of a tree. This can be
formally described as follows. Let $#codes_in$ be a set of "inner nodes", and let $#codes' = #codes
union #codes_in$. Let $#tree$ be a tree with nodes $#codes'$ and whose leaves are all and only the
codes $#codes$. This tree is called _ontology_ and gives some informations over the codes in $#codes$
that appeares in the dataset. We will assume that codes that are closer with respect to the distance
on the graph $#tree$ to be more similar than codes that are less close.

As done in @panigutti-xai we proceed to describe a distance on the possible patients $#patients$. To
do that, we will use a distance over the codes $#codes$ and the visits $#visits$.

We start describing a code-to-code distance. It is the Wu-Palmer similarity score, which is one of
the most commonly used for the ICD ontology. Let $c_1$ and $c_2$ be two codes in $#codes$. Let $L$
be their lowest common ancestor on the tree $#tree$, and let $R$ be the root of $#tree$. Let $d(c_1',
c_2')$ be the distance between codes in $#codes'$ which measures the smallest number of steps needed
to reach $c_2'$ starting from $c_1'$ and moving along the edges of the graph $#tree$. the Wu-Palmer
similarity score is then defined as
$ "WuP"(c_1, c_2) = (2 d(L,R)) / (d(c_1,L) + d(c_2, L) + 2d(L,R)) $

We observe that $0 <= "WuP"(c_1, c_2) <= 1$ for each pair of codes, and the minimum value $0$ is
reached when $L$ is the root of the tree, while the maximum value $1$ is reached when $c_1 = c_2 =
L$.

We can then define a visit-to-visit distance. The approach of @panigutti-xai is to use an edit
distance weighted through the Wu-Palmer similarity. However would need to choose an order between
the codes of each visit, while the codes do not have a natural order. For this reason we will follow
a different approach. Let $V_A = {c^A_1, ..., c^A_a}$ and $V_B = {c^B_1,..., c^B_b}$ be two visits
composed by $a$ and $b$ codes respectively. For each code in $V_A$ we choose the best similar code
in $V_B$. We them sum all of the distances between the best pairs to get an asymetric distance:
$ d^#visits _"asym" (V_A, V_B) = sum_(i=1)^a min_(j=1...b) "WuP"(v^A_i, v^B_j). $
We can symmetrize the above expression taking the maximum of the two permutations:
$ d^#visits  (V_A, V_B) = max(d^#visits _"asym" (V_A, V_B), d^#visits _"asym" (V_B, V_A)). $

Assuming to have precomputed a table with all the distance pairs $"Wup"(c_1, c_2)$, the computation
of $d^#visits (V_A, V_B)$ has a complexity of $cal("O")(n^2)$, where $n$ is an upper limit on the
size of the visits.

Finally we are ready to describe a patient-to-patient distance $d^#patients$. We do that through
the Dynamic Time Warp (DTW) Algorithm. It gives a measure of similarity between two time series that can
differ in speed. The idea is to find associations between two elements of each series, in our
context they are visits of two patients, subject to the following constraints:
- Every element of the first sequence must be associated to an element of the first one;
- The first and the last elements of each sequence must be associated between them, but they not to
  be their only association.
- The associations must be monotonical: Let the $i$-th element of the first sequence is associated
  to the $j$-th element of the second one, and a similar things happens for the $i'$-th element in
  the first sequence and the $j'$-th element of the second one. Then $i<i'$ implies $j<= j'$.
We can give a cost to each way of associating two series: it is the sum of the visit-to-visit
distances between each associated pair. The DTW similarity is then defined as the minimum cost
between all the associations that respect the previous conditions.

This optimization problem can be solved with a dynamic programming approach. Here it follows the
algorithm in pseudocode:
#note("Spudoratamente preso da Wikipedia")
```
int DTWDistance(s: array [1..n], t: array [1..m]) {
    DTW := array [0..n, 0..m]
    
    for i := 0 to n
        for j := 0 to m
            DTW[i, j] := infinity
    DTW[0, 0] := 0
    
    for i := 1 to n
        for j := 1 to m
            cost := d(s[i], t[j])
            DTW[i, j] := cost + minimum(DTW[i-1, j  ],    // insertion
                                        DTW[i  , j-1],    // deletion
                                        DTW[i-1, j-1])    // match
    
    return DTW[n, m]
}
```
#note("Posso spiegare più in dettaglio qui")

It should be noted that the measure we have defined is not a distance in the sense of the metric
spaces, in fact it cannot guarantee the triangular inequality. Moreover if the number of visit per
patient is bounded by $m$ and the number of codes per visit is bounded by $n$, then there will be
$m^2$ iterations within the DTW algorithm, each of those will require a computation of the
visit-to-visit distance, which costs $cal("O")(n^2)$ iterations. Thus the total cost of the
algorithm is $cal("O")(m^2n^2)$.

== Solution Proposal

= Implementation of the Solution

= Experiments

== Mimic IV

Mimic IV @mimic-iv-cit is a dataset contaning medical informations resulted from the collaboration
of the Beth Israeli Deaconess Medical Center (BIDMC) and the Massachusetts Institute of Technology
(MIT). Data is gathered as part of the routine activities at BIDMC, and processed at MIT.

The dataset contains informations about patients accessing the services at the emergency department
or Intensive Care Units (ICUs) between 2008 and 2019. Patients that were below age 18 at ther first
admission time were excluded. People known to require extra protection were excluded as well.
Between the raw data sources and the final published dataset, a step of deintefication of the
patients has been performed, removing all personal informations and translating by a random time all
the events regarding every single patient.

The dataset is divided into three separate modules: `hosp`, `icu`, and `note`.

The `hosp` module stores information regarding patient transfers, billed events, medication
prescription, medication administration, laboratory values, microbiology measurements, and provider
orders. This is the main source of data of this project.

The `icu` module contains all the information collected by the MetaVision clinical information
system for the ICU units.

The `note` module contains textual data of discharges, which gathers an in-depth summary of the
patients history during their stays at the hospital, and a section about radiology data.

Our main interest is in the `hosp` module, which contains among other things a table of admissions
and a table of diagnoses. The admissions is a table that associates a patient id, an admission id,
several temporal coordinates for the admissions such as the admission, registration and discharge
times. There are also information about the patient like language, insurance, race and marital
status. We will use only the patient and admission id, and the relative ordering given by the
admission time. This table contains 431,231 rows, each corresponding to a unique admission. 

The diagnoses table is much larger, containing 4,756,326 rows, each with a unique diagnose. A
diagnose is composed by a patient id and admission id, which correspond to the ones given in the
admission table, a couple icd-version and icd-code, and a numerical priority ranging 1-39 which
ranks the importance of the codes within the same visit.

ICD, the International Classification of Diseases, which is a system to classify diagnostic
statements mantained by the World Health Organizaton (WHO). There are several versions of these
codes, and Mimic-iv uses ICD-9 and ICD-10. The version is flagged in a specific column of the
diagnoses table. A code may indicate signs, symptoms, abnormal findings, complaints, social
circumstances, and external causes of injury or disease. In @icd-examples are reported some examples
of ICD-10 codes. ICD-9 codes are very similar.

#figure(
table(
  columns: (auto, auto),
  [Code], [Description],
  [A00.0], [Cholera due to Vibrio cholerae 01, biovar cholerae],
  [E66.0], [Obesity due to excess calories],
  [I22.0], [Subsequent myocardial infarction of anterior wall],
  [F32.0], [Mild depressive episode],
  [O80.0], [Spontaneous vertex delivery],
  [S81.0], [Open wound of knee],
  [W58],   [Bitten or struck by crocodile or alligator],
  [Z56.3], [Stressful work schedule],
  [Z72.0], [Tobacco use (Excl. tobacco dependence)],
),
caption: [Examples of ICD-10 codes with their descriptions.]
) <icd-examples>

In the diagnoses table of the Mimic-iv dataset there are 2,766,877 ICD-9 codes (58% of the total)
and 1,989,449 codes (42% of the total). Considering codes with different versions of ICD
categorization as different, the dataset contains 25,829 distinct codes, which means that each code
appears on average 184 times in the database. Of course the codes are not evenly distributed among
those present. #note("Grafico della distribuzione").

The admission table contains 180,733 distinct patients. However there are 101,198 patients (56% of
the total) who are present in a single admission. We will filter those out because to make a
prediction that can be compared with a ground truth at least two visits are necessary. A percentage
of 92% of patients has no more than five visits.

A preprocess phase is done by deleting all the patients who have a single visit, and then a
conversion from ICD-9 to ICD-10 is performed, such that all the models will work on a single
ontology.

== Analysis

= Conclusions
