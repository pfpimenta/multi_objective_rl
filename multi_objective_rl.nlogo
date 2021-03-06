; Parameters:
; grid_size : size of the grid (grid_size X grid_size)
; max_steps_per_episode :
; num_episodes :
; exploration_rate :
; learning_algorithm :

extensions [matrix py]

; defining grid size (min and max x,y values in distance from the origin 0,0 )
globals [
  min_x
  min_y
  max_x
  max_y
  was_captured ; True if the prey was captured in this step
  ticks-per-episode
  run-number
]


; preys and predators are both breeds of turtles
breed [ preys prey ]  ; preys is its own plural, so we use "a-preys" as the singular
breed [ predators predator ]

preys-own [action ]
predators-own [
  action
  reward-proximity
  reward-angle
  reward-separation
  state
  last-state
  q-table-proximity
  q-table-angle
  q-table-separation
]


to setup
  print "\nenvironment setup."
  clear-all

  py:setup py:python
  py:run "import numpy as np"
  py:run "from ensemble_functions import majority_voting, ranking_voting"

  setup-grid-size
  create-agents
  set was_captured False
  ask predators [update-state]
  reset-ticks
end

to create-agents

  ; remove old predators and prey
  ask turtles
  [die]

  create-preys 1  ; create the preys, then initialize their variables
  [
    set shape "sheep"
    set color white
    set size 1.5  ; easier to see
    set label-color blue - 2
    set-random-coords
  ]

  create-predators 2  ; create the predators, then initialize their variables
  [
    set shape "wolf"
    set color red
    set size 2  ; easier to see
    set-random-coords

    set q-table-proximity init-q-table
    set q-table-angle init-q-table
    set q-table-separation init-q-table
  ]
end

; to be called before a new run begins
to reset-run
  create-agents
  set was_captured False
  ask predators [
    update-state
    set last-state state
  ]
end

; to be called before a new episode begins
to reset-episode
  set was_captured False
  reset-positions
  reset-ticks
end

to set-random-coords
  let x-coordinate random grid_size
  let y-coordinate random grid_size
  setxy x-coordinate y-coordinate
end

to setup-grid-size
  set min_x 0 - (grid_size / 2)
  set min_y 0 - (grid_size / 2)
  set max_x (grid_size / 2)
  set max_y (grid_size / 2)
  resize-world min_x max_x min_y max_y
end

to-report init-q-table
  let n-rows num-states
  let n-cols 5; num actions : do-nothing, move-up, move-down, move-left, move-right
  let initialValue 0.01
  let table (matrix:make-constant n-rows n-cols initialValue)
  report table
end

to-report num-states
  ; grid_size_plus_one possible distances (pred-2 and prey, x and y), counting with 0
  let grid_size_plus_one (grid_size + 1)
  report grid_size_plus_one  * grid_size_plus_one  * grid_size_plus_one  * grid_size_plus_one
end

to go
  print "\nStarting simulation..."
  run-runs
  print "\nEnd."
  stop
end

to run-runs
  set run-number 0
  loop [
    reset-run
    set run-number (run-number + 1)
    type "\nStarting run " type run-number print "..."
    run-episodes
    save-results
    if run-number = number-of-runs [
      stop
    ]
  ]
end

to run-episodes
  set ticks-per-episode []
  let episode-number 0
  loop [
    set episode-number (episode-number + 1)
    type "Starting episode " type episode-number print "..."
    simulation-episode
    set ticks-per-episode lput ticks ticks-per-episode
    if episode-number = num_episodes [
      print ticks-per-episode
      stop
    ]
  ]
end

to save-results

  let ticks-filename (word "outputs/" ensemble_algorithm "/ticks_per_episode/ticks_per_episode_" run-number ".txt")
  type "Saving " type ticks-filename print "..."
  file-open ticks-filename
  file-write ticks-per-episode
  file-close

  let q-tables-filename (word "outputs/" ensemble_algorithm "/q_tables/q_tables_" run-number ".txt")
  type "Saving " type q-tables-filename print "..."
  file-open q-tables-filename
  ask predators [
    file-write "predator q-table-proximity... "
    file-write q-table-proximity
    file-write "predator q-table-angle... "
    file-write q-table-angle
    file-write "predator q-table-separation... "
    file-write q-table-separation
  ]
  file-close
end

to simulation-episode
  reset-positions
  set was_captured False
  reset-ticks
  loop [
    if simulation-step [
      type "...ended after " type ticks print " ticks with the prey captured. (predators won)"
      stop
    ]
    if ticks = max_steps_per_episode [
      type "...ended after " type ticks print " ticks without the prey captured. (prey won)"
      stop
    ]
  ]
end

to reset-positions
  ask preys
    [set-random-coords]
  ask predators [set-random-coords]
  ask predators [update-state]
end

to-report simulation-step
  ; choose actions
  choose-actions
  ; execute actions
  execute-actions
  ; check if end condition was reached
  ifelse check-end-condition
    [set was_captured True]
    [set was_captured False]

  ask predators [set-reward]

  ; update q-tables
  ask predators [update-q-tables]

  tick
  report was_captured
end

to choose-actions
  ask preys
  [choose-action-prey]
  ask predators
  [choose-action-predator]
end

to execute-actions
  ask preys
  [move]
  ask predators
  [move]
end

to choose-action-prey
  ifelse random-float 1 < 0.2
    [set action random 5]
    [choose-action-move-away-from-predators]
end

to choose-action-predator
  ifelse random-float 1 < (exploration_rate / 100)
    [set action random 5] ; exploration
    [choice-action-predator-exploitation] ; exploitation
;    [choice-action-predator-hard-coded] ; for debugging
end

to choice-action-predator-exploitation
  update-state
  let current-row-proximity (matrix:get-row q-table-proximity state)
  py:set "row_proximity" current-row-proximity
  let current-row-angle (matrix:get-row q-table-angle state)
  py:set "row_angle" current-row-angle
  let current-row-separation (matrix:get-row q-table-separation state)
  py:set "row_separation" current-row-separation

  if ensemble_algorithm = "majority_voting_ensemble"
  [
    set action py:runresult "majority_voting(row_proximity, row_angle, row_separation)"
  ]
  if ensemble_algorithm = "ranking_voting_ensemble"
  [
    set action py:runresult "ranking_voting(row_proximity, row_angle, row_separation)"
  ]
end

to update-state
  set last-state state
  let pred1-x xcor
  let pred1-y ycor
  let pred2-x (reduce + ([xcor] of predators)) - pred1-x
  let pred2-y (reduce + ([ycor] of predators)) - pred1-y
  let prey-x reduce + ([xcor] of preys)
  let prey-y reduce + ([ycor] of preys)

  ; compute state hash
  let state-0 min list (int (pred1-x - pred2-x + grid_size)) grid_size
  let state-1 min list (int (pred1-y - pred2-y + grid_size)) grid_size
  let state-2 min list (int (pred1-x - prey-x + grid_size)) grid_size
  let state-3 min list (int (pred1-y - prey-y + grid_size)) grid_size
;  type "debug state: " type state-0 type " " type state-1 type " " type state-2 type " " print state-3
  let grid_size_plus_one (grid_size + 1)
  set state (state-0 + (grid_size_plus_one  * state-1) + (grid_size_plus_one  * grid_size_plus_one  * state-2) + (grid_size_plus_one  * grid_size_plus_one  * grid_size_plus_one  * state-3))
end

to choice-action-predator-hard-coded
  ; moves closer to the prey

  ; compute on the values (distance from predators) of each action
  let predator_x xcor
  let predator_y ycor
  ; value of moving up
  let distance-list [distancexy predator_x (predator_y + 1)] of preys
  let up-value reduce + distance-list
  ; value of moving down
  set distance-list [distancexy predator_x (predator_y - 1)] of preys
  let down-value reduce + distance-list
  ; value of moving left
  set distance-list [distancexy (predator_x - 1) predator_y] of preys
  let left-value reduce + distance-list
  ; value of moving right
  set distance-list [distancexy (predator_x + 1) predator_y] of preys
  let right-value reduce + distance-list
  ; value of not moving
  set distance-list [distancexy predator_x predator_y] of preys
  let dont-move-value reduce + distance-list

  ; choose action based on the values (summed distance from predators) of each action
  set action 0 ; by default, dont move
  let value grid_size * grid_size * grid_size; just to declare the variable with a really big value
  let values-list (list dont-move-value up-value down-value left-value right-value)
  let min-value min values-list
  let best-move position min-value values-list

  set action best-move
end

to set-reward
  let base-reward 0
  if was_captured
  [set base-reward (base-reward + 1)]

  set reward-proximity (base-reward + proximity-shaping)
  set reward-angle (base-reward + angle-shaping)
  set reward-separation (base-reward + separation-shaping)
end

to-report proximity-shaping
  ; proximity shaping value (negative manhatan distance form prey)
  let dist-list [abs (xcor - [xcor] of myself) + abs (ycor - [ycor] of myself)] of preys
  let proximity_shaping (-1 * (reduce + dist-list))
  if normalize_shapings [
    report (proximity_shaping / (2 * grid_size))
  ]
  report proximity_shaping
end

to-report angle-shaping
  ; angle shaping value
  ; arccos((vector-pred1-prey * vector-pred2-prey) / (abs(vector-pred1-prey)* abs(vector-pred2-prey)))
  let pred1-x xcor
  let pred1-y ycor
  let pred2-x (reduce + ([xcor] of predators)) - pred1-x
  let pred2-y (reduce + ([ycor] of predators)) - pred1-y
  let prey-x reduce + ([xcor] of preys)
  let prey-y reduce + ([ycor] of preys)
  let dist-pred1-prey-x (abs prey-x - pred1-x)
  let dist-pred1-prey-y (abs prey-y - pred1-y)
  let dist-pred2-prey-x (abs prey-x - pred2-x)
  let dist-pred2-prey-y (abs prey-y - pred2-y)
  let abs-dist-pred1-prey (sqrt (dist-pred1-prey-x * dist-pred1-prey-x + dist-pred1-prey-y * dist-pred1-prey-y))
  let abs-dist-pred2-prey (sqrt (dist-pred2-prey-x * dist-pred2-prey-x + dist-pred2-prey-y * dist-pred2-prey-y))
  let vec-prod (dist-pred1-prey-x * dist-pred2-prey-x + dist-pred1-prey-y * dist-pred2-prey-y)
  let angle 0
  ifelse (abs-dist-pred1-prey * abs-dist-pred2-prey) = 0
  [set angle 1]
  [
    let acos-param (vec-prod / (abs-dist-pred1-prey * abs-dist-pred2-prey))
    set angle (acos (max (list -1 (min (list 1 acos-param)))))
  ]
  if normalize_shapings [
    let approximated-pi 3.14159265
    report (angle / approximated-pi)
  ]
  report angle
end

to-report separation-shaping
  ;  separation shaping value
  let dist-list [abs (xcor - [xcor] of myself) + abs (ycor - [ycor] of myself)] of predators
  let separation  (max dist-list)
  if normalize_shapings [
    report (separation / (2 * grid_size))
  ]
  report separation
end

to-report linear-scalarization-shaping
  ; define shaping weights
  let a 0.3
  let b 0.3
  let c 0.3
  ; get shapings
  let proximity proximity-shaping
  let angle angle-shaping
  let separation separation-shaping
  ; aggregation with linear scalarization
  let linear_scalarization (a * proximity + b * angle + c * separation)
  report linear_scalarization
end

to choose-action-move-away-from-predators
  ; compute on the values (distance from predators) of each action
  let prey_x xcor
  let prey_y ycor
  ; value of moving up
  let distance-list [distancexy prey_x (prey_y + 1)] of predators
  let up-value reduce + distance-list
  ; value of moving down
  set distance-list [distancexy prey_x (prey_y - 1)] of predators
  let down-value reduce + distance-list
  ; value of moving left
  set distance-list [distancexy (prey_x - 1) prey_y] of predators
  let left-value reduce + distance-list
  ; value of moving right
  set distance-list [distancexy (prey_x + 1) prey_y] of predators
  let right-value reduce + distance-list
  ; value of not moving
  set distance-list [distancexy prey_x prey_y] of predators
  let dont-move-value reduce + distance-list

  ; choose action based on the values (summed distance from predators) of each action
  set action 0 ; by default, dont move
  let value grid_size * grid_size * grid_size; just to declare the variable with a really big value
  let values-list (list dont-move-value up-value down-value left-value right-value)
  let max-value max values-list
  let best-move position max-value values-list

  set action best-move
end

to move
  if action = 0
    [dont-move]
  if action = 1
    [move-up]
  if action = 2
    [move-down]
  if action = 3
    [move-left]
  if action = 4
    [move-right]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;
to update-q-tables
  update-q-table-proximity
  update-q-table-angle
  update-q-table-separation
end

to update-q-table-proximity
  let old-value (matrix:get q-table-proximity last-state action)
  let current_row (matrix:get-row q-table-proximity state)
  let optimal-future-value (max current_row)
  let temporal_difference (reward-proximity + (discount_factor * optimal-future-value) - old-value)
  let new-value (old-value + (learning_rate * temporal_difference))
  matrix:set q-table-proximity last-state action new-value
end

to update-q-table-angle
  let old-value (matrix:get q-table-angle last-state action)
  let current_row (matrix:get-row q-table-angle state)
  let optimal-future-value (max current_row)
  let temporal_difference (reward-angle + (discount_factor * optimal-future-value) - old-value)
  let new-value (old-value + (learning_rate * temporal_difference))
  matrix:set q-table-angle last-state action new-value
end

to update-q-table-separation
  let old-value (matrix:get q-table-separation last-state action)
  let current_row (matrix:get-row q-table-separation state)
  let optimal-future-value (max current_row)
  let temporal_difference (reward-separation + (discount_factor * optimal-future-value) - old-value)
  let new-value (old-value + (learning_rate * temporal_difference))
  matrix:set q-table-separation last-state action new-value
end

;;;;;;;;;;;;;;;;;;;;;;;;;;
; checks if end condition was reached
to-report check-end-condition
  let has_ended False
  ask predators [
    if any? preys with [abs (xcor - [xcor] of myself) < 1 and abs (ycor - [ycor] of myself) < 1]
      [set has_ended True]
  ]
  report has_ended
end

; returns True if agent is in the same position as the coords passed by parameter
to-report check-same-position [pos_x pos_y]
  ifelse xcor = pos_x or ycor = pos_y
      [report True]
      [report False]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;
; actions:
; 0 dont-move
; 1 move-up
; 2 move-down
; 3 move-left
; 4 move-righ

to move-up
  ; check if agent would move past the border
  ifelse ycor > max_y - 1
      [dont-move]
      [set ycor ycor + 1]
end

to move-down
  ; check if agent would move past the border
  ifelse ycor < min_y + 1
      [dont-move]
      [set ycor ycor - 1]
end

to move-left
  ; check if agent would move past the border
  ifelse xcor < min_x + 1
      [dont-move]
      [set xcor xcor - 1]
end

to move-right
  ; check if agent would move past the border
  ifelse xcor > max_x - 1
      [dont-move]
      [set xcor xcor + 1]
end

to dont-move
  ; do not do anything
end
@#$#@#$#@
GRAPHICS-WINDOW
240
10
694
465
-1
-1
21.24
1
14
1
1
1
0
1
1
1
-10
10
-10
10
1
1
1
ticks
30.0

BUTTON
20
25
89
80
setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
105
25
160
80
go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

INPUTBOX
20
95
177
155
grid_size
20.0
1
0
Number

SLIDER
20
290
217
323
exploration_rate
exploration_rate
0
100
10.0
1
1
%
HORIZONTAL

CHOOSER
20
395
215
440
ensemble_algorithm
ensemble_algorithm
"majority_voting_ensemble" "ranking_voting_ensemble"
0

INPUTBOX
20
160
175
220
max_steps_per_episode
5000.0
1
0
Number

INPUTBOX
120
225
215
285
num_episodes
2000.0
1
0
Number

INPUTBOX
20
330
110
390
learning_rate
0.1
1
0
Number

INPUTBOX
120
330
215
390
discount_factor
0.995
1
0
Number

SWITCH
20
445
217
478
normalize_shapings
normalize_shapings
0
1
-1000

INPUTBOX
20
225
110
285
number-of-runs
10.0
1
0
Number

@#$#@#$#@
## WHAT IS IT?

This model explores the stability of predator-prey ecosystems. Such a system is called unstable if it tends to result in extinction for one or more species involved.  In contrast, a system is stable if it tends to maintain itself over time, despite fluctuations in population sizes.

## HOW IT WORKS

There are two main variations to this model.

In the first variation, the "sheep-wolves" version, wolves and sheep wander randomly around the landscape, while the wolves look for sheep to prey on. Each step costs the wolves energy, and they must eat sheep in order to replenish their energy - when they run out of energy they die. To allow the population to continue, each wolf or sheep has a fixed probability of reproducing at each time step. In this variation, we model the grass as "infinite" so that sheep always have enough to eat, and we don't explicitly model the eating or growing of grass. As such, sheep don't either gain or lose energy by eating or moving. This variation produces interesting population dynamics, but is ultimately unstable. This variation of the model is particularly well-suited to interacting species in a rich nutrient environment, such as two strains of bacteria in a petri dish (Gause, 1934).

The second variation, the "sheep-wolves-grass" version explictly models grass (green) in addition to wolves and sheep. The behavior of the wolves is identical to the first variation, however this time the sheep must eat grass in order to maintain their energy - when they run out of energy they die. Once grass is eaten it will only regrow after a fixed amount of time. This variation is more complex than the first, but it is generally stable. It is a closer match to the classic Lotka Volterra population oscillation models. The classic LV models though assume the populations can take on real values, but in small populations these models underestimate extinctions and agent-based models such as the ones here, provide more realistic results. (See Wilensky & Rand, 2015; chapter 4).

The construction of this model is described in two papers by Wilensky & Reisman (1998; 2006) referenced below.

## HOW TO USE IT

1. Set the model-version chooser to "sheep-wolves-grass" to include grass eating and growth in the model, or to "sheep-wolves" to only include wolves (black) and sheep (white).
2. Adjust the slider parameters (see below), or use the default settings.
3. Press the SETUP button.
4. Press the GO button to begin the simulation.
5. Look at the monitors to see the current population sizes
6. Look at the POPULATIONS plot to watch the populations fluctuate over time

Parameters:
MODEL-VERSION: Whether we model sheep wolves and grass or just sheep and wolves
INITIAL-NUMBER-SHEEP: The initial size of sheep population
INITIAL-NUMBER-WOLVES: The initial size of wolf population
SHEEP-GAIN-FROM-FOOD: The amount of energy sheep get for every grass patch eaten (Note this is not used in the sheep-wolves model version)
WOLF-GAIN-FROM-FOOD: The amount of energy wolves get for every sheep eaten
SHEEP-REPRODUCE: The probability of a sheep reproducing at each time step
WOLF-REPRODUCE: The probability of a wolf reproducing at each time step
GRASS-REGROWTH-TIME: How long it takes for grass to regrow once it is eaten (Note this is not used in the sheep-wolves model version)
SHOW-ENERGY?: Whether or not to show the energy of each animal as a number

Notes:
- one unit of energy is deducted for every step a wolf takes
- when running the sheep-wolves-grass model version, one unit of energy is deducted for every step a sheep takes

There are three monitors to show the populations of the wolves, sheep and grass and a populations plot to display the population values over time.

If there are no wolves left and too many sheep, the model run stops.

## THINGS TO NOTICE

When running the sheep-wolves model variation, watch as the sheep and wolf populations fluctuate. Notice that increases and decreases in the sizes of each population are related. In what way are they related? What eventually happens?

In the sheep-wolves-grass model variation, notice the green line added to the population plot representing fluctuations in the amount of grass. How do the sizes of the three populations appear to relate now? What is the explanation for this?

Why do you suppose that some variations of the model might be stable while others are not?

## THINGS TO TRY

Try adjusting the parameters under various settings. How sensitive is the stability of the model to the particular parameters?

Can you find any parameters that generate a stable ecosystem in the sheep-wolves model variation?

Try running the sheep-wolves-grass model variation, but setting INITIAL-NUMBER-WOLVES to 0. This gives a stable ecosystem with only sheep and grass. Why might this be stable while the variation with only sheep and wolves is not?

Notice that under stable settings, the populations tend to fluctuate at a predictable pace. Can you find any parameters that will speed this up or slow it down?

## EXTENDING THE MODEL

There are a number ways to alter the model so that it will be stable with only wolves and sheep (no grass). Some will require new elements to be coded in or existing behaviors to be changed. Can you develop such a version?

Try changing the reproduction rules -- for example, what would happen if reproduction depended on energy rather than being determined by a fixed probability?

Can you modify the model so the sheep will flock?

Can you modify the model so that wolves actively chase sheep?

## NETLOGO FEATURES

Note the use of breeds to model two different kinds of "turtles": wolves and sheep. Note the use of patches to model grass.

Note use of the ONE-OF agentset reporter to select a random sheep to be eaten by a wolf.

## RELATED MODELS

Look at Rabbits Grass Weeds for another model of interacting populations with different rules.

## CREDITS AND REFERENCES

Wilensky, U. & Reisman, K. (1998). Connected Science: Learning Biology through Constructing and Testing Computational Theories -- an Embodied Modeling Approach. International Journal of Complex Systems, M. 234, pp. 1 - 12. (The Wolf-Sheep-Predation model is a slightly extended version of the model described in the paper.)

Wilensky, U. & Reisman, K. (2006). Thinking like a Wolf, a Sheep or a Firefly: Learning Biology through Constructing and Testing Computational Theories -- an Embodied Modeling Approach. Cognition & Instruction, 24(2), pp. 171-209. http://ccl.northwestern.edu/papers/wolfsheep.pdf .

Wilensky, U., & Rand, W. (2015). An introduction to agent-based modeling: Modeling natural, social and engineered complex systems with NetLogo. Cambridge, MA: MIT Press.

Lotka, A. J. (1925). Elements of physical biology. New York: Dover.

Volterra, V. (1926, October 16). Fluctuations in the abundance of a species considered mathematically. Nature, 118, 558???560.

Gause, G. F. (1934). The struggle for existence. Baltimore: Williams & Wilkins.

## HOW TO CITE

If you mention this model or the NetLogo software in a publication, we ask that you include the citations below.

For the model itself:

* Wilensky, U. (1997).  NetLogo Wolf Sheep Predation model.  http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation.  Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

Please cite the NetLogo software as:

* Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

## COPYRIGHT AND LICENSE

Copyright 1997 Uri Wilensky.

![CC BY-NC-SA 3.0](http://ccl.northwestern.edu/images/creativecommons/byncsa.png)

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.  To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/3.0/ or send a letter to Creative Commons, 559 Nathan Abbott Way, Stanford, California 94305, USA.

Commercial licenses are also available. To inquire about commercial licenses, please contact Uri Wilensky at uri@northwestern.edu.

This model was created as part of the project: CONNECTED MATHEMATICS: MAKING SENSE OF COMPLEX PHENOMENA THROUGH BUILDING OBJECT-BASED PARALLEL MODELS (OBPML).  The project gratefully acknowledges the support of the National Science Foundation (Applications of Advanced Technologies Program) -- grant numbers RED #9552950 and REC #9632612.

This model was converted to NetLogo as part of the projects: PARTICIPATORY SIMULATIONS: NETWORK-BASED DESIGN FOR SYSTEMS LEARNING IN CLASSROOMS and/or INTEGRATED SIMULATION AND MODELING ENVIRONMENT. The project gratefully acknowledges the support of the National Science Foundation (REPP & ROLE programs) -- grant numbers REC #9814682 and REC-0126227. Converted from StarLogoT to NetLogo, 2000.

<!-- 1997 2000 -->
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.0
@#$#@#$#@
set model-version "sheep-wolves-grass"
set show-energy? false
setup
repeat 75 [ go ]
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
1
@#$#@#$#@
