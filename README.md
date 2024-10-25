# AI_expo_bargaining
- Implementation of OpenSpiel bargaining game in python
## Game overview
- 2 player game
- Agents negotiate on how to split a pool of objects
- There are 3 different types of objects (call them a,b,c)
- In total there are 5-7 objects
- The agents are each assigned a randomly generated valuation for each type of object (e.g 2 points for a, 1 for b, 0 for c)
- The total valuation for all the objects in the pool will always be 10 for both agents
- The agents take turn making offers on how to split the objects
- e.g pool: 3, 2 ,1, (3 objects of type a, 2 of type b and 1 of type c) agent 1 offers (2,1,0) (agent 1 takes 2 a, 1 b, 0 c) leaving (1,1,1) for agent 2
- Agents can accept the offer or make a counter offer until 10 rounds have passed after which if no deal is made, then both agents get nothing
- Reward is simply the final value of objects obtained by the agent
## Dependencies
- https://github.com/google-deepmind/open_spiel/blob/master/docs/install.md
