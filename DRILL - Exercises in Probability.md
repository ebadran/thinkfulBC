
### Unit 3 Lesson 2 Project 4
### DRILL - Exercises in Probability

#### Q1:
Calculate the probability of flipping a balanced coin four times and getting each pattern: HTTH, HHHH and TTHH.

P(red ∩ blue) = P(red) \* P(blue) = (1/2) \* (1/2) = 1/4


```python
1/2 * 1/2 * 1/2 * 1/2
```




    0.0625



#### Q2:
If a list of people has 24 women and 21 men, then the probability of choosing a man from the list is 21/45. What is the probability of not choosing a man?


```python
24/45
```




    0.5333333333333333



#### Q3:
The probability that Bernice will travel by plane sometime in the next year is 10%. The probability of a plane crash at any time is .005%. What is the probability that Bernice will be in a plane crash sometime in the next year?

The two variables are inconditional, therefore:
P(red ∩ blue) = P(red) \* P(blue) = (1/2) \* (1/2) = 1/4


```python
.00005 * .1
```




    5e-06



#### Q4:
A data scientist wants to study the behavior of users on the company website. Each time a user clicks on a link on the website, there is a 5% chance that the user will be asked to complete a short survey about their behavior on the website. The data scientist uses the survey data to conclude that, on average, users spend 15 minutes surfing the company website before moving on to other things. What is wrong with this conclusion?

**Answer:** If a user spends more time on the website, there's a greater chance that they will be asked to complete the survey. The correct procedure would be to ask 5% of all users that *when they enter* the website.
