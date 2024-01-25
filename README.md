# Extended Framework for ML models in Streaming Anomaly Detection
This is a complimentary repository for the publication [Insert publication].

## Fundamental Tasks
A streaming anomaly detection algorithm is formalized to consist of 4 fundamental tasks:
- Data representation
- Learning strategy
- Nonconformity score
- Anomaly score

## Code Design
Every method for a fundamental task is implemented as a class according to an abstract class of that task. 
Thereby, a method includes one publisher and multiple subscribers. Instances of methods for different tasks are envisioned to be connected as
Publisher <--> Subscriber according to the Observer pattern. In order to retain a low code complexity, instantiation and connection of fundamental 
task methods is done only in main.py.
