let allGames = [];
let fuse = null; // Holds the fuzzy search engine
let selectedGameId = null;
let searchTimeout = null;

// 1. Load all games and initialize Fuse.js
async function loadGames() {
    try {
        const response = await fetch('/games');
        allGames = await response.json();
        console.log(`Loaded ${allGames.length} games.`);

        // Initialize Fuse with settings
        fuse = new Fuse(allGames, {
            keys: ['name'],       // Search inside the 'name' field
            threshold: 0.3,       // 0.0 = perfect match, 1.0 = match anything. 0.3 is good for typos.
            ignoreLocation: true, // Matches "Catan" anywhere in the string
            minMatchCharLength: 2 // Don't search for single letters
        });

    } catch (error) {
        console.error('Failed to load games:', error);
    }
}

// 2. Debounce handler
function handleInput() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        filterGames();
    }, 300);
}

// 3. Fuzzy Filter Logic
function filterGames() {
    const gameName = document.getElementById('gameName').value.trim();
    const searchResults = document.getElementById('searchResults');

    if (!gameName) {
        searchResults.innerHTML = '';
        selectedGameId = null;
        document.getElementById('selectedGame').innerHTML = '';
        return;
    }

    // Perform Fuzzy Search
    const results = fuse.search(gameName).slice(0, 10);

    // Render results
    if (results.length > 0) {
        searchResults.innerHTML = '<ul>' +
            results.map(result => {
                // Fuse returns the object inside an 'item' property
                const game = result.item;
                const safeName = game.name.replace(/'/g, "\\'");
                return `<li onclick="selectGame('${game.id}', '${safeName}')">${game.name}</li>`;
            }).join('') + '</ul>';
    } else {
        searchResults.innerHTML = '<p>No games found</p>';
    }
}

// 4. Handle Selection
function selectGame(gameId, gameName) {
    selectedGameId = gameId;
    document.getElementById('gameName').value = gameName;
    document.getElementById('searchResults').innerHTML = '';
    document.getElementById('selectedGame').innerHTML = `Selected: ${gameName}`;
}

// 5. Get Recommendations
async function getRecommendations() {
    const resultsDiv = document.getElementById('results');

    if (!selectedGameId) {
        resultsDiv.innerHTML = '<p>Please search and select a game first</p>';
        return;
    }

    resultsDiv.innerHTML = '<p>Loading recommendations...</p>';

    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game_id: selectedGameId
            })
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.detail || 'Failed to get recommendations');

        if (data.length > 0) {
            resultsDiv.innerHTML = '<h3>Recommendations:</h3><ul>' +
                data.map(game =>
                    `<li>${game.name} (Score: ${game.match_score.toFixed(2)})</li>`
                ).join('') + '</ul>';
        } else {
            resultsDiv.innerHTML = '<p>No recommendations found</p>';
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
    }
}

// 6. Initialization
document.addEventListener('DOMContentLoaded', function() {
    loadGames();

    const gameNameInput = document.getElementById('gameName');
    gameNameInput.addEventListener('input', handleInput);
    gameNameInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && selectedGameId) {
            getRecommendations();
        }
    });
});
