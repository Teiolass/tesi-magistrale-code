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
a special `<start>` token, the model generates the next token. 

A recurrent model is built in such a way that it can take in input a hidden memory vector $h$, and a
token $t$, and then outputs $h'$, an updated version of $h$. The new hidden vector is then fed to
the model together with the next token. This process is repeated 

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
the $i$-th component of $h(p, theta)$. For example the vector $x=(x_1, dots, x_(|cal("C")|)})$ assigns
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
