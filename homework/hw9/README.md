------
# Fall 2018 IE534/CS598:  HW9

@author: Ziyu Zhou

------


>There are three sets of results for comparison: 1.) single-frame model, 2.) 3D model, 3.) combined output of the two models. For each of these three, report the following:
>
>- (top1_accuracy,top5_accuracy,top10_accuracy): Did the results improve after combining the outputs?
>- Use the confusion matrices to get the 10 classes with the highest performance and the 10 classes with the lowest performance (_Report 10 high performing classes and 10 poor performing classes (max and min of the diagonal elements)_): Are there differences/similarities? Can anything be said about whether particular action classes are discriminated more by spatial information versus temporal information?
>- Use the confusion matrices to get the 10 most confused classes (_Report 10 pairs of classes which are the off-diagonal elements (most confused)_). That is, which off-diagonal elements of the confusion matrix are the largest: Are there any notable examples?
>
>Put all of the above into a report and submit as a pdf. Also zip all of the code (not the models, predictions or dataset) and submit.

## Accuracies

|                    | Top 1 Accuracy | Top 5 Accuracy | Top 10 Accuracy |
| ------------------ | -------------- | -------------- | --------------- |
| Single-frame model | 0.779540       | 0.945282       | 0.979381        |
| 3D model           | 0.833466       | 0.970923       | 0.984404        |
| Combined model     | 0.867301       | 0.978588       | 0.989162        |

The results improve after combining the outputs. All accuracies increase compared to both the single frame model and the 3D model, especiallty the `top1_accuracy` which increases from ~78% in the single-frame model to ~87% in the combined model.





<div style="page-break-after: always;"></div>



## Best/Worst Performing Classes

Class with highest performance should be a class with most samples being classified correctly. Conversely, class with lowest performance should be a class with least samples being classified correctly.

Note that the value in the parentheses shows the probability of a particular class being correctly identified based on the `confusion_matrix`.

|                    | 10 Classes with the Highest Performance                      | 10 Classes with the Lowest Performance                       |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Single-frame model | Rowing (1.0),  Surfing (1.0), BasketballDunk (1.0), Billiards(1.0), PlayingDaf (1.0), PlayingTabla (1.0), HorseRace (1.0), RockClimbingIndoor (1.0), FrisbeeCatch (1.0), Skijet (1.0) | JumpRope (0.03), BodyWeightSquats (0.07), YoYo (0.19), JumpingJack (0.22), HandstandWalking (0.23), Shotput (0.28), HighJump (0.32), Nunchucks (0.4), CricketShot (0.43), BrushingTeeth (0.44) |
| 3D model           | Drumming (1.0), JumpRope (1.0), Fencing (1.0), HorseRiding (1.0), PoleVault (1.0), PlayingViolin (1.0), PlayingTabla (1.0), PlayingPiano (1.0), PlayingGuitar (1.0), BoxingSpeedBag (1.0) | CricketShot (0.22), HandstandWalking (0.29),  Lunges (0.30), HighJump (0.35), Nunchucks (0.37), YoYo (0.47), FrontCrawl (0.49), SoccerJuggling (0.51), ApplyLipstick (0.53), PommelHorse (0.54) |
| Combined model     | JumpingJack (1.0), Rafting (1.0), PlayingDhol (1.0), BenchPress (1.0), BasketballDunk (1.0), Bowling (1.0), ParallelBars (1.0), BoxingSpeedBag (1.0), RopeClimbing (1.0), PlayingViolin (1.0) | HandstandWalking (0.32), HighJump (0.32), Lunges (0.51),  Nunchucks (0.51), YoYo (0.53), CricketShot (0.53), BrushingTeeth (0.56), FrontCrawl (0.6), PizzaTossing (0.61), Haircut (0.61) |

There are some classes that can be easily recognized by the 3D model but hard for the single-frame model to classify. For example, the probability of _JumpRope_ being correctly classified by the single-frame model is just `0.03`, while this probability is `1.0` for the 3D model. This means that _JumpRope_ requires more temporal information to be correctly identified than spatial information.

On the contrary, some classes may require more spatial information than temporal information. _GolfSwing_, although it's not among the best 10/worst 10 in the table, has a probability of `0.78` by the single-frame model and `0.58` by the 3D model.

The overal performance of the three models is **combined model > 3D model > single frame model**. This makes sense as the combined model combines the spatial and temporal information together and thus making it more precise. _YoYo_ has only `0.19` probability of correctness using the single frame model. The 3D model makes it to `0.47` and the combined model finally increases it to `0.53`.

There are also classes that can be easily classified by each of the three models such as classes including _playing_ something like _PlayingViolin_.



<div style="page-break-after: always;"></div>



## Most Confused Classes

|                    | 10 Most Confused Classes                                     |
| ------------------ | ------------------------------------------------------------ |
| Single-frame model | <ul><li>('BrushingTeeth', 'ShavingBeard')</li><li>('ApplyEyeMakeup', 'ApplyLipstick')</li> <li>('HighJump', 'JavelinThrow')</li> <li>('FrontCrawl', 'BreastStroke')</li> <li>('BodyWeightSquats', 'Lunges')</li> <li>('JumpRope', 'HulaHoop')</li> <li>('CricketShot', 'CricketBowling')</li> <li>('Shotput', 'ThrowDiscus')</li> <li>('Haircut', 'BlowDryHair')</li> <li>('Hammering', 'HeadMassage')</li></ul> |
| 3D model           | <ul><li>('CricketShot', 'CricketBowling')</li> <li>('FrontCrawl', 'BreastStroke')</li> <li>('Haircut', 'BlowDryHair')</li> <li>('PommelHorse', 'ParallelBars')</li> <li>('Kayaking', 'Rafting')</li> <li>('ApplyLipstick', 'ApplyEyeMakeup')</li> <li>('HammerThrow', 'ThrowDiscus')</li> <li>('Lunges', 'FloorGymnastics')</li> <li>('BrushingTeeth', 'ApplyEyeMakeup')</li> <li>('HighJump', 'PoleVault')</li></ul> |
| Combined model     | <ul><li>('FrontCrawl', 'BreastStroke')</li> <li>('PommelHorse', 'ParallelBars')</li> <li>('Haircut', 'BlowDryHair')</li> <li>('Nunchucks', 'TaiChi')</li> <li>('BrushingTeeth', 'ShavingBeard')</li> <li>('CricketShot', 'CricketBowling')</li> <li>('HammerThrow', 'ThrowDiscus')</li> <li>('HighJump', 'JavelinThrow')</li> <li>('Kayaking', 'Rafting')</li> <li>('HighJump', 'PoleVault')</li></ul> |

The three models have very similar confusing classes.

For all three models, for example, the pairs of ('CricketShot', 'CricketBowling'), ('Haircut', 'BlowDryHair'), ('FrontCrawl', 'BreastStroke') are very confusing.