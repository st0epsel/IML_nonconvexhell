1. Task description

This task is about logistic regression: given an input vector $x$, your goal is to predict the value of a binary random variable $y$ where the logits of $y$ can be modelled as a linear function of a set of feature transformations, $\phi(x)$. In other words, the labels $Y$ given $X$ can be modelled using a logistic regression model where the inputs are a feature-transformed $X$.
Data description

In the handout for this project, you will find the the following files:

- train.csv: the training set
- sample.csv: a sample submission file in the correct format
- template_solution.py: a template file that will guide you through the implementation of the solution
- template_solution.ipynb: a template file in Jupyter notebook format that will guide you through the implementation of the solution

You are free to use either jupyter notebook or the .py template file.

Each line in train.csv represents one data instance by an id, its label $y$, and its features $x_1$ to $x_5$:

```csv
Id,y,x1,x2,x3,x4,x5
0,1,0.019999999999999907,0.04999999999999993,-0.09000000000000008,-0.43000000000000005,-0.08000000000000007
...
```

Features description

You are required to use the following features (in the following order) to make your predictions:

- Linear: $\phi_1(x)=x_1$, $\phi_2(x)=x_2$, $\phi_3(x)=x_3$, $\phi_4(x)=x_4$, $\phi_5(x)=x_5$
- Quadratic: $\phi_6(x)=x_1^2$, $\phi_7(x)=x_2^2$, $\phi_8(x)=x_3^2$, $\phi_9(x)=x_4^2$, $\phi_{10}(x)=x_5^2$
- Exponential: $\phi_{11}(x)=e^{x_1}$, $\phi_{12}(x)=e^{x_2}$, $\phi_{13}(x)=e^{x_3}$, $\phi_{14}(x)=e^{x_4}$, $\phi_{15}(x)=e^{x_5}$
- Cosine: $\phi_{16}(x)=\cos(x_1)$, $\phi_{17}(x)=\cos(x_2)$, $\phi_{18}(x)=\cos(x_3)$, $\phi_{19}(x)=\cos(x_4)$, $\phi_{20}(x)=\cos(x_5)$
- Constant: $\phi_{21}(x)=1$

where we indicate the whole input vector with $x$ and we use $x_i$ to denote its $i$th component.

Your predictions model the logits of y as a linear function of the features above according to the following formula:

$$
\hat{P}(y=1\mid x)=\sigma\left(\sum_{j=1}^{21} w_j\phi_j(x)\right)
$$

We provide a template solution file that suggests a structure for how you can solve the task, by filing in the TODOs in the skeleton code. It is not mandatory to use this solution template but it is recommended since it should make getting started on the task easier. You are also encouraged (but not required) to implement logistic regression solutions from scratch, for a deeper understanding of the course material.
Submission format

You are required to submit the weights of your linear predictor in a .csv file.

The file should contain 21 lines containing a float each. The i-th line indicates the i-th weight of your linear predictor. For your convenience, we further provide a sample submission file:

```text
1
2
...
```

Notice that, to compute your prediction on the test data, the raw features of the test data are transformed according to the transformations introduced in the previous section and their dot products with your submitted weight vector are computed, before taking the sigmoid of the scalar result to produce a probability. This means that the first entry of your weight vector is multiplied by $\phi_1(x)$, the second entry is multiplied by $\phi_2(x)$ and so on. As a consequence, it is important to submit the weight vector in the correct order.

Please keep in mind that, as a group, you have a limited number of submissions as stated on the submissions page.
Evaluation

The evaluation metric for this task is the F1 Score, which is the harmonic mean of precision and recall. This metric balances the model's ability to correctly identify whether a positive instance is positive (Precision) and its ability to classify all positive instances as positive (Recall).

$$
F_1 = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
$$

Where:

- $\mathrm{Precision}=\frac{TP}{TP+FP}$
- $\mathrm{Recall}=\frac{TP}{TP+FN}$

We abbreviated True Positives (TP), False Positives (FP) and False Negatives (FN) in the formulas above.

To compute these, your continuous probability estimates $\hat{y}_i = \hat{P}(y=1\mid x_i)$ are converted by us into hard labels (0 or 1) using a threshold of 0.5. Your goal is to maximize the $F_1$ score, i.e., achieve the best balance between precision and recall.
Grading

In this task you will submit a weight vector. We compute the performance of the resulting predictor on a test set. When handing in the task, you need to select which of your submissions will get graded and provide a short video description of your approach. This has to be done individually by each member of the team. You achieve a pass (6.0) if you achieve a better score than the single baseline, and a fail (2.0) otherwise. For the pass/fail decision, we also consider the code and the video submission explaining your solution. The following non-binding guidance provides you with an idea on what is expected to pass the project: If you hand in a proper video submission, your source code is runnable and reproduces your submitted csv, and your submission performs better than the baseline, you can expect to have passed the assignment. 