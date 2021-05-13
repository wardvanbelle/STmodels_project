# Overview
## 1 Purpose
This model is created to analyze the effects of certain parameters (pedestrian density, stampede location, evacuation strategy,...) on the efficiency
of an evacuation and the resulting victim count.
## 2 Entities, state variables and scales
* Entities: 
  * Agents: Pedestrians
  * Local environment: A room enclosed by walls with one exit in the center of the left wall.
  * Global environment: None?
* Pedestrian state variables: 
  * Location
  * Preferred strategy
    * S1: Stay away from stampede location
    * S2: Follow movement of neighbours
    * S3: Follow a chosen direction until reaching a wall and evacuate following the wall
  * Movement state
    * U_e: Unaffected by stampede and can perceive exit
    * U_n: Unaffected by stampede and can't perceive exit
    * A_E: Affected by stampede and can perceive exit
    * A_n: Affected by stampede and can't perceive exit
    * C: Fallen down
* Scales:
  * Temporal scale: Each time step represents the time it takes for a pedestrian to go to another cell, and the model is evaluated for 100 time steps.
It is assumed to be 0.3 s, implying a walking speed of 1.33 m/s
  * Spatial scale: Every cell is 0.4 x 0.4 m<sup>2</sup>, and the room is 12 x 12 m<sup>2</sup>, resulting in a 30 x 30 cell grid
## 3 Process overview and scheduling 
(Description of all processes executed by the entities (pedestrians) + the order in which they're executed in pseudo-code)
# Design concepts
## Basic principles
