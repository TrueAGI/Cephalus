# Cephalus


---

## Project Status

Cephalus is currently very much a work in progress. There's a lot to be done 
before it's release-ready or even usable at all, frankly. Watch this space for 
major progress updates, keep an eye on TODO.md for minor progress updates, or 
say hi if you're looking for a way to contribute.

A few places I can **definitely** use some help:
* Documentation
* Unit tests
* Design discussion

By all means, open an issue or submit a pull request if you want to jump
straight in!

---


## Introduction

Cephalus is a NOS (Neural OS). It serves as a way to harness the use of
multiple sensors and a learned internal state representation to accomplish 
tasks of fixed or indefinite duration within a shared environment. Tasks 
can offer feedback to the system to hone the state representation. Because
the environment is shared among tasks, each task can benefit from the
improvements to the state representation made at the behest of other tasks.


### A Note on Adaptation of Policy Gradients for State Representation Induction

Any effective policy gradient method can be easily adapted to induce a 
state representation for reinforcement learning. Simply treat the negative
of the prediction loss as a reward signal in its own right, and use the
'action' produced by the policy gradient as a representation of the state
history, appending it to the externally sourced inputs fed to the algorithm
on the next time step.

#### A Long-Winded Example

Suppose we have a reinforcement learning problem with partially hidden state.
Old school Snake, for example, for which the motion of the snake cannot be 
determined from any single video frame. (Technically, Snake does not have hidden 
state, but rather has state information which is distributed out over multiple 
inputs, requiring memory. But who is bothering to split that hair? Oops. I'll 
stop.)

Now Snake has a discrete action space. The player just pushes one of the 
direction arrows and the snake will begin moving in that direction. That 
means we can use a standard N-armed bandit to estimate the action Q values.
(Which is why I picked it for this example, to reduce confusion.) Let's 
pretend for the snake of discussion that we need a state representation for 
this game, and that for some undetermined reason we are not able to keep
a history of recent video frames to feed to our Q value model. Whatever are 
we to do?

Well, let's figure out a way to compress the necessary historical 
information into a single, small, continuous vector space. We'll just
provide the most recent video frame and this so-called state vector to
our Q value model on each step. Then we'll create a new state vector for 
the next time step by compressing the last video frame and state vector
together into that same small vector space.

'Okay,' you say. 'How do we pick the best way to compress the data together 
into that state vector, so that we keep only the most relevant information
around and forget the other stuff that would get in our way?' (Remember, 
we're pretending here.) Well, there are tons of ways to potentially approach
that problem, but the way we are interested in is to use reinforcement
learning.

That's right. *Another* reinforcement learning agent, showing up to assist 
its fellow RL agent in trouble, by providing useful historical information to 
it in the form of a state vector. RL agents, unite!

Now I mentioned earlier that this state vector is continuous. Which means
we need a reinforcement learning algorithm that handles continuous action
spaces. That's where policy gradients come in. Let's pick an arbitrary one.
I'll go with AAC (Advantage Actor-Critic), because I like how dramatic it 
sounds when you pronounce the acronym as if it were a word. **AAC!** Ahem.

Okay, what should we use as the reward for this secondary RL agent?
Well, we want this second agent to choose a state representation that
helps the first one out. We have a couple of choices here. We can pass the
same reward that the first agent receives to the second one. Or,
we can take a look at how *accurate* the Q value estimates of the first
agent are, and try to optimize that. For reasons that I will explain
shortly, we'll go with the second option, and try to maximize the accuracy
of the first agent's Q value estimates, rather than directly maximizing 
the same reward signal as the first agent. The idea here is that we
want the second agent to learn to pick a state representation that helps 
the first one make *educated* decisions.

How do we maximize the accuracy of the first agent's Q value estimates?
Well, we can just measure its prediction losses -- the distance between
the predicted Q value for an action and the training target for that Q 
value produced when we have more information, i.e. the actual reward
received on the next time step. The lower the prediction loss, the better
the second agent is doing its job of informing the first one. So we just 
negate the prediction loss of the first agent and take that as our reward 
signal for the second one.

What about inputs to the second agent to help it make good decisions?
It seems intuitive that the same information that the first agent
needs to make good decisions during its game play will also be useful to
the second agent to make good decisions about what to remember for the 
next step. So lets just feed it the same inputs as the first agent:
the latest video frame, and the latest state vector. Yeah, we are feeding
the actions of the second agent directly back to it as inputs. Talk
about eating your words...

Now here's where it gets cool: Because we use temporal differences to 
propagate reward signals from each game play compression decision to the 
one made before it, our second RL agent will actually learn the best 
state representation not just for the next time step, but for multiple 
time steps in advance. So that really complicated game of Snake? Don't 
worry, we can learn to remember all the useful info we might need for 
later decisions in the game, and how to cram all that *unbelievably rich* 
data down into one small state vector.

So hopefully now its clear as a bell how we can use RL agents to
solve RL agent problems. Teamwork makes the dream work!

Oh, yeah. I promised I would explain why we optimize for the first 
agent's prediction accuracy instead of using the same reward. There
are a couple of reasons.

First, think about this: The information that's useful for making one 
prediction has a good chance of being useful for making other 
predictions about the same situation. If you know which way the snake is 
moving, it'll help you assess whether it's on course to get the next 
nibble of food, and it'll also help you assess whether it's on course to 
slam into a wall. And the first agent, the one that picks the direction 
the snake is going, is making multiple predictions on every time step -- 
one per direction key. Supposing our first agent hasn't realized yet that 
always turning right to avoid impending doom of smashing at high snake 
speeds into a wall isn't necessarily the best strategy. Well, thanks to
its buddy, agent #2, it has the information it needs to learn that turning
left occasionally isn't a bad idea, without even realizing it needs it yet.
And that's because agent #2 isn't as biased about which information it feeds
to agent #1, as agent #1 is about which choice it makes once it has that
information.

The second reason is actually closely related to the first one. Let's
imagine that there's actually something else to do in Snake land besides
chasing food and dodging stuff. And we've made yet *another* RL agent
which learns to do that other, more interesting stuff. Because we have
optimized for *accuracy of prediction*, agent #2 can hop right in and
start helping out our new fella, too. Of course, we might need to do a
little pre-training of our new guy, and maybe some reward normalization,
too, before agent #2 starts actually paying attention to the new reward
signals, but the state representations that agent #2 has already been
trained to provide will serve as an excellent information source to get
the new guy up to speed ASAP.

So there you have it! 
