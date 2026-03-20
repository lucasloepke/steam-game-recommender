	The issue is that the Steam gaming library is so vast and diverse that many users may feel lost about which games to play next; not wanting to waste money or time on a game they won’t enjoy, however not knowing which games they will enjoy. The goal of our project is to solve this problem by developing a game recommending system for Steam users using collaborative filtering. It will take a given user’s existing library and predict which game the user is most likely to enjoy. 
Formally, let U = {u1, u2, …, um} denote the set of users and G = {g1, g2, …, gn}denote the set of games. We define an interaction matrix M ∈ R(m×n), where each entry Mij represents the interaction between user ui and game gj. Two formulations will be considered:
1. Binary formulation:
Mij =    { 1 if user ui owns or has played game gj
{ 0 otherwise
2. Implicit feedback formulation (hours played):
Mij = hours played by user ui on game gj
If Mij = 0, we assume no interaction.
The matrix M is sparse, as each user interacts with only a small subset of all available games.
The learning task is to approximate M with a low-rank factorization: M ≈ PQT , where P ∈ R(m×k)
represents latent user features and Q ∈ R(n×k) represents latent game features. The prediction task is to estimate missing entries Mij for games not yet played by user ui, and recommend the top-k games with the highest predicted scores.
	Our dataset is sourced from the Steam Web API and consists of user libraries. Each entry will include the user ID, game ID, and hours played. Consider preprocessing the training data, due to the size of the data set, it would be more efficient to filter out games with low frequency played and users with minimal games in their library. IDK what else to say about the data set

“progress we’ve made” 😭

The final report outline: 
Introduction
Our problem and solution
Importance of our project
Related Work
Other recommender systems
What is collaborative filtering
Mothodology
Formulas and algorithms
Dataset and Preprocessing
Our dataset
Testing and Results
Our model and evaluation
Conclusion/Discussion
Strengths and weaknesses
Future steps

tentative next steps
