# Implemented a new enhanced model file at models/HPG.py without touching models/novel_model.py.

---
                                                                                                     
### What is added in models/HPG.py to push beyond small gains:   

#### 1. SegmentPooler

- Builds fixed hierarchical segments from long token sequences (pseudo document-level structure).   

#### 2. SalienceAwarePlanner
- Scores segment salience.
- Uses learned plan queries to extract multiple plan tokens from salient segments.
- Refines plan tokens with transformer layers.


####  3. PlanConditionedFusion 
- Lets encoder token states attend to plan tokens and fuse them through a learned gate.
- Produces plan-aware encoder states before decoding.


#### 4. Auxiliary planning objectives            
                                                                                                            
- planner_entropy term (focuses salience distribution).
- plan_redundancy penalty (reduces repetitive plan tokens).
- Added to generation loss with configurable weights.