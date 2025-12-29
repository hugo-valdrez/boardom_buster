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

-> add feedback system. (did oyu like this game? yes/no. why no?)

-> if the user clicks on a game recommended, it should link the user to the bgg page.

-> add radar data, it looks cool and it's interactive.

-> mention the reason why the game was picked: most similar (its the most similar game), most popular (it's a similar and popular game), it's a good game (good rating)
