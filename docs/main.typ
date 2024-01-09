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
suitable size.

As in most deep learning models, skip connections and layer normalizations are also added. #note("Di
più?")

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

== Solution Proposal

= Implementation of the Solution

= Experiments

== Mimic IV

== Analysis

= Conclusions
