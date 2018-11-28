------
# Fall 2018 IE534/CS598:  HW6

@author: Ziyu Zhou

------


>There are three sets of results for comparison: 1.) single-frame model, 2.) 3D model, 3.) combined output of the two models. For each of these three, report the following:
>
>- (top1_accuracy,top5_accuracy,top10_accuracy): Did the results improve after combining the outputs?
>- Use the confusion matrices to get the 10 classes with the highest performance and the 10 classes with the lowest performance: Are there differences/similarities? Can anything be said about whether particular action classes are discriminated more by spatial information versus temporal information?
>- Use the confusion matrices to get the 10 most confused classes. That is, which off-diagonal elements of the confusion matrix are the largest: Are there any notable examples?
>
>Put all of the above into a report and submit as a pdf. Also zip all of the code (not the models, predictions or dataset) and submit.

## Accuracies

|                    | Top 1 Accuracy | Top 5 Accuracy | Top 10 Accuracy |
| ------------------ | -------------- | -------------- | --------------- |
| Single-frame model | 0.779540       | 0.945282       | 0.979381        |
| 3D model           |                |                |                 |
| Combined model     |                |                |                 |



## Best/Worst Performing Classes

Class with highest performance should be a class with most samples being classified correctly. Conversely, class with lowest performance should be a class with least samples being classified correctly.

Note that the value in the parentheses shows the probability of a particular class being correctly identified based on the `confusion_matrix`.

|                    | 10 Classes with the Highest Performance                      | 10 Classes with the Lowest Performance                       |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Single-frame model | Rowing (1.0),  Surfing (1.0), BasketballDunk (1.0), Billiards(1.0), PlayingDaf (1.0), PlayingTabla (1.0), HorseRace (1.0), RockClimbingIndoor (1.0), FrisbeeCatch (1.0), Skijet (1.0) | JumpRope (0.03), BodyWeightSquats (0.07), YoYo (0.19), JumpingJack (0.22), HandstandWalking (0.23), Shotput (0.28), HighJump (0.32), Nunchucks (0.4), CricketShot (0.43), BrushingTeeth (0.44) |
| 3D model           |                                                              |                                                              |
| Combined model     |                                                              |                                                              |



## Most Confused Classes

|                    | 10 Most Confused Classes |
| ------------------ | ------------------------ |
| Single-frame model |                          |
| 3D model           |                          |
| Combined model     |                          |

