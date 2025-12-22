to do:
-> the user might want to change the weights of the recommendation system.

-> add a field regarding complexity (iirc it was the weight on the bgg dataset)

-> maybe used the min age as weight?

-> use a "golden set" to tune the weights.

-> vectorized dataset names, to quick search + possibly remove games with the same name (Catan vs Catan expansions, dont want to recommend only catan expansions). 18/12/2025
    -> Let's use fuzzy wording instead. 21/12/2025

-> encode the game description and use it as a vector to test similarity. 18/12/2025 (on-hold)
    -> embedding runs ok, the problem is calculating distances. 19/12/2025 (on-hold)
    -> also tested it with the game names, same thing. 19/12/2025 (on-hold)