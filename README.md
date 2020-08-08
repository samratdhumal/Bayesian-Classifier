# Bayesian-Classifier

### The goal of this project is to construct a classifier such that for any given values of 𝐹1 and 𝐹2, it can predict the performed task (𝐶1, 𝐶2, C3, C4, 𝐶5). We used the powerful Bayes Theorem for classification.

In an experiment involving 1000 participants, we recorded two different measurement (F1 and F2) while participants performed 5 different tasks (C1, C2, ..., C5). The two measurements are independent and for each class they  can be considered to have a normal distribution as follow:

P (F1 | Ci) = N (m1i, σ21i) and P (F2 | Ci) = N (m2i, σ22i)
for i = 1, 2, … ,5
where m1i, σ21i are the mean and variance of F1 for the ith class and m2i, σ22i are the mean and variance   of F2 for the ith class.

Using Bayes Theorem to build a Naive Bayes classifier to calculate the probability of each class given the measurement data, and output the most probable class as the predicted class.
