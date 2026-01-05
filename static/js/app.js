/**
 * Boardom Buster - Board Game Recommendation App
 * Interactive UI with fuzzy search and radar chart visualizations
 */

// ========================================
// State Management
// ========================================
let allGames = [];
let fuse = null;
let selectedGame = null;
let searchTimeout = null;

// DOM Elements
const searchInput = document.getElementById('searchInput');
const autocompleteDropdown = document.getElementById('autocompleteDropdown');
const excludeFamilyToggle = document.getElementById('excludeFamilyToggle');
const instructions = document.getElementById('instructions');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const selectedGameImage = document.getElementById('selectedGameImage');
const selectedGameName = document.getElementById('selectedGameName');
const recommendationsList = document.getElementById('recommendationsList');
const errorMessage = document.getElementById('errorMessage');

// ========================================
// Initialization
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    loadGames();
    setupEventListeners();
});

function setupEventListeners() {
    // Search input events
    searchInput.addEventListener('input', handleSearchInput);
    searchInput.addEventListener('focus', handleSearchFocus);
    searchInput.addEventListener('keydown', handleSearchKeydown);

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
            hideDropdown();
        }
    });
}

// ========================================
// Data Loading
// ========================================
async function loadGames() {
    try {
        const response = await fetch('/games');
        allGames = await response.json();
        console.log(`Loaded ${allGames.length} games.`);

        // Initialize Fuse.js for fuzzy search
        fuse = new Fuse(allGames, {
            keys: ['name'],
            threshold: 0.3,
            ignoreLocation: true,
            minMatchCharLength: 2
        });
    } catch (error) {
        console.error('Failed to load games:', error);
        showError('Failed to load game database. Please refresh the page.');
    }
}

// ========================================
// Search & Autocomplete
// ========================================
function handleSearchInput() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        performSearch();
    }, 200);
}

function handleSearchFocus() {
    if (searchInput.value.trim()) {
        performSearch();
    }
}

function handleSearchKeydown(e) {
    const items = autocompleteDropdown.querySelectorAll('.autocomplete-item');
    const activeItem = autocompleteDropdown.querySelector('.autocomplete-item.active');
    let currentIndex = Array.from(items).indexOf(activeItem);

    switch (e.key) {
        case 'ArrowDown':
            e.preventDefault();
            if (currentIndex < items.length - 1) {
                if (activeItem) activeItem.classList.remove('active');
                items[currentIndex + 1].classList.add('active');
                items[currentIndex + 1].scrollIntoView({ block: 'nearest' });
            }
            break;
        case 'ArrowUp':
            e.preventDefault();
            if (currentIndex > 0) {
                if (activeItem) activeItem.classList.remove('active');
                items[currentIndex - 1].classList.add('active');
                items[currentIndex - 1].scrollIntoView({ block: 'nearest' });
            }
            break;
        case 'Enter':
            e.preventDefault();
            if (activeItem) {
                const gameId = activeItem.dataset.gameId;
                const gameName = activeItem.dataset.gameName;
                selectGame(gameId, gameName);
            } else if (selectedGame) {
                getRecommendations();
            }
            break;
        case 'Escape':
            hideDropdown();
            break;
    }
}

function performSearch() {
    const query = searchInput.value.trim();

    if (!query) {
        hideDropdown();
        return;
    }

    if (!fuse) {
        console.warn('Search not ready yet');
        return;
    }

    const results = fuse.search(query).slice(0, 10);

    if (results.length > 0) {
        renderDropdown(results);
        showDropdown();
    } else {
        autocompleteDropdown.innerHTML = `
            <div class="autocomplete-item" style="cursor: default; color: #666;">
                No games found
            </div>
        `;
        showDropdown();
    }
}

function renderDropdown(results) {
    autocompleteDropdown.innerHTML = results.map((result, index) => {
        const game = result.item;
        const thumbnail = game.thumbnail_url || '';
        return `
            <div class="autocomplete-item ${index === 0 ? 'active' : ''}"
                 data-game-id="${game.id}"
                 data-game-name="${escapeHtml(game.name)}"
                 onclick="selectGame('${game.id}', '${escapeHtml(game.name).replace(/'/g, "\\'")}')">
                ${thumbnail
                    ? `<img class="autocomplete-item-img" src="${thumbnail}" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="autocomplete-item-icon" style="display:none;">🎲</div>`
                    : `<div class="autocomplete-item-icon">🎲</div>`
                }
                <div class="autocomplete-item-name">${escapeHtml(game.name)}</div>
            </div>
        `;
    }).join('');
}

function showDropdown() {
    autocompleteDropdown.classList.add('active');
}

function hideDropdown() {
    autocompleteDropdown.classList.remove('active');
}

// ========================================
// Game Selection
// ========================================
function selectGame(gameId, gameName) {
    selectedGame = { id: gameId, name: gameName };
    searchInput.value = gameName;
    hideDropdown();

    // Automatically get recommendations
    getRecommendations();
}

// ========================================
// Recommendations
// ========================================
async function getRecommendations() {
    if (!selectedGame) {
        showError('Please search and select a game first');
        return;
    }

    // Show loading state
    showLoadingState();

    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game_id: parseInt(selectedGame.id),
                top_k: 5,
                exclude_same_family: excludeFamilyToggle.checked
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }

        // Add minimum delay to show loading animation
        await new Promise(resolve => setTimeout(resolve, 1500));

        if (data.recommendations && data.recommendations.length > 0) {
            renderResults(data);
        } else {
            showError('No recommendations found for this game');
        }
    } catch (error) {
        console.error('Error getting recommendations:', error);
        showError(error.message || 'An error occurred while getting recommendations');
    }
}

// ========================================
// UI State Management
// ========================================
function showLoadingState() {
    hideError();
    instructions.classList.add('hidden');
    resultsSection.classList.remove('active');
    loadingSection.classList.add('active');
}

function showResultsState() {
    loadingSection.classList.remove('active');
    instructions.classList.add('hidden');
    resultsSection.classList.add('active');
}

function showError(message) {
    loadingSection.classList.remove('active');
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

// ========================================
// Render Results
// ========================================
function renderResults(data) {
    const { input_game, recommendations } = data;

    // Update selected game display with input game's info
    selectedGameName.textContent = input_game.name;

    if (input_game.image) {
        selectedGameImage.src = input_game.image;
        selectedGameImage.onerror = () => {
            selectedGameImage.src = 'data:image/svg+xml,' + encodeURIComponent(getPlaceholderSvg());
        };
    } else {
        selectedGameImage.src = 'data:image/svg+xml,' + encodeURIComponent(getPlaceholderSvg());
    }

    // Render selected game card with full info
    renderSelectedGameCard(input_game);

    // Render recommendation cards
    recommendationsList.innerHTML = recommendations.map((game, index) =>
        createRecommendationCard(game, index)
    ).join('');

    // Draw radar charts
    recommendations.forEach((game, index) => {
        drawRadarChart(`radar-${index}`, game);
    });

    showResultsState();
}

function renderSelectedGameCard(game) {
    const selectedGameCard = document.getElementById('selectedGameCard');
    const categories = game.categories || [];
    const mechanics = game.mechanics || [];
    const description = game.description || '';
    const bggLink = game.bgg_link || '';

    selectedGameCard.innerHTML = `
        <img class="selected-game-image"
             id="selectedGameImage"
             src="${game.image || 'data:image/svg+xml,' + encodeURIComponent(getPlaceholderSvg())}"
             alt="${escapeHtml(game.name)}"
             onerror="this.src='data:image/svg+xml,' + encodeURIComponent(getPlaceholderSvg())">
        <div class="selected-game-name">${escapeHtml(game.name)}</div>
        ${description ? `<div class="selected-game-description">${escapeHtml(truncateText(description, 200))}</div>` : ''}
        ${categories.length > 0 ? `
            <div class="selected-game-meta">
                <div class="selected-game-meta-label">Categories:</div>
                <div class="selected-game-tags">
                    ${categories.map(cat => `<span class="game-tag">${escapeHtml(cat)}</span>`).join('')}
                </div>
            </div>
        ` : ''}
        ${mechanics.length > 0 ? `
            <div class="selected-game-meta">
                <div class="selected-game-meta-label">Mechanics:</div>
                <div class="selected-game-tags">
                    ${mechanics.map(mech => `<span class="game-tag">${escapeHtml(mech)}</span>`).join('')}
                </div>
            </div>
        ` : ''}
        ${bggLink ? `
            <a href="${bggLink}" target="_blank" rel="noopener noreferrer" class="bgg-button selected-game-bgg">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                    <polyline points="15,3 21,3 21,9"/>
                    <line x1="10" y1="14" x2="21" y2="3"/>
                </svg>
                View on BGG
            </a>
        ` : ''}
    `;
}

function createRecommendationCard(game, index) {
    const isHighlighted = index === 0;
    const categories = game.categories || [];
    const mechanics = game.mechanics || [];
    const imageUrl = game.image || '';
    const description = game.description || '';
    const comment = game.comment || '';

    return `
        <div class="recommendation-card ${isHighlighted ? 'highlighted' : ''}">
            <div class="game-image-container">
                <img class="game-image"
                     src="${imageUrl || 'data:image/svg+xml,' + encodeURIComponent(getPlaceholderSvg())}"
                     alt="${escapeHtml(game.name)}"
                     onerror="this.src='data:image/svg+xml,' + encodeURIComponent(getPlaceholderSvg())">
            </div>

            <div class="game-info">
                <div class="game-name">${escapeHtml(game.name)}</div>
                ${description ? `<div class="game-description">${escapeHtml(truncateText(description, 200))}</div>` : ''}
                ${comment ? `<div class="game-comment">${escapeHtml(comment)}</div>` : ''}

                <div class="game-meta">
                    ${categories.length > 0 ? `
                        <div class="game-meta-row">
                            <span class="game-meta-label">Categories:</span>
                            <span class="game-meta-value">
                                ${categories.slice(0, 4).map(cat => `<span class="game-tag">${escapeHtml(cat)}</span>`).join('')}
                            </span>
                        </div>
                    ` : ''}
                    ${mechanics.length > 0 ? `
                        <div class="game-meta-row">
                            <span class="game-meta-label">Mechanics:</span>
                            <span class="game-meta-value">
                                ${mechanics.slice(0, 4).map(mech => `<span class="game-tag">${escapeHtml(mech)}</span>`).join('')}
                            </span>
                        </div>
                    ` : ''}
                </div>
            </div>

            <div class="game-actions">
                <div class="radar-container">
                    <canvas id="radar-${index}" class="radar-chart" width="120" height="120"></canvas>
                </div>

                <div class="metrics-list">
                    ${createMetricRow('Sim', game.cosine_similarity, 'How similar the game mechanics and categories are')}
                    ${createMetricRow('Diff', game.difficulty_similarity, 'How close the difficulty levels are')}
                    ${createMetricRow('Time', game.playing_time_similarity, 'How similar the playing time is')}
                    ${createMetricRow('Rating', game.avg_rating, 'Community average rating')}
                    ${createMetricRow('Pop', game.popularity, 'Number of ratings/reviews')}
                </div>

                <a href="${game.bgg_link}" target="_blank" rel="noopener noreferrer" class="bgg-button">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                        <polyline points="15,3 21,3 21,9"/>
                        <line x1="10" y1="14" x2="21" y2="3"/>
                    </svg>
                    View on BGG
                </a>
            </div>
        </div>
    `;
}

function createMetricRow(label, value, tooltip) {
    const percentage = ((value || 0) * 100).toFixed(0);
    return `
        <div class="metric-row">
            <span class="metric-item-tooltip">
                <span class="metric-label">${label}</span>
                <span class="metric-help">?</span>
                <span class="metric-help-text">${tooltip}</span>
            </span>
            <span class="metric-value">${percentage}%</span>
        </div>
    `;
}

// ========================================
// Radar Chart Drawing
// ========================================
function drawRadarChart(canvasId, game) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const size = 140;
    canvas.width = size;
    canvas.height = size;

    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size / 2 - 30;

    // Data points (normalized 0-1)
    const data = [
        game.cosine_similarity || 0,
        game.difficulty_similarity || 0,
        game.playing_time_similarity || 0,
        game.avg_rating || 0,
        game.popularity || 0
    ];

    const labels = ['Sim', 'Diff', 'Time', 'Rating', 'Pop'];
    const numPoints = data.length;
    const angleStep = (2 * Math.PI) / numPoints;
    const startAngle = -Math.PI / 2;

    // Clear canvas
    ctx.clearRect(0, 0, size, size);

    // Draw background pentagon layers
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;

    for (let level = 0.2; level <= 1; level += 0.2) {
        ctx.beginPath();
        for (let i = 0; i < numPoints; i++) {
            const angle = startAngle + i * angleStep;
            const x = centerX + radius * level * Math.cos(angle);
            const y = centerY + radius * level * Math.sin(angle);
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.closePath();
        ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#ccc';
    for (let i = 0; i < numPoints; i++) {
        const angle = startAngle + i * angleStep;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(
            centerX + radius * Math.cos(angle),
            centerY + radius * Math.sin(angle)
        );
        ctx.stroke();
    }

    // Draw data polygon
    ctx.fillStyle = 'rgba(51, 51, 51, 0.2)';
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;

    ctx.beginPath();
    for (let i = 0; i < numPoints; i++) {
        const angle = startAngle + i * angleStep;
        const value = Math.max(0.1, data[i]); // Minimum value for visibility
        const x = centerX + radius * value * Math.cos(angle);
        const y = centerY + radius * value * Math.sin(angle);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Draw data points
    ctx.fillStyle = '#333';
    for (let i = 0; i < numPoints; i++) {
        const angle = startAngle + i * angleStep;
        const value = Math.max(0.1, data[i]);
        const x = centerX + radius * value * Math.cos(angle);
        const y = centerY + radius * value * Math.sin(angle);

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Draw labels
    ctx.fillStyle = '#666';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < numPoints; i++) {
        const angle = startAngle + i * angleStep;
        const labelRadius = radius + 18;
        const x = centerX + labelRadius * Math.cos(angle);
        const y = centerY + labelRadius * Math.sin(angle);
        ctx.fillText(labels[i], x, y);
    }
}

// ========================================
// Utility Functions
// ========================================
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength).trim() + '...';
}

function getPlaceholderSvg() {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect width="100" height="100" fill="#f0e6d3"/>
        <rect x="25" y="25" width="50" height="50" rx="5" fill="none" stroke="#333" stroke-width="3"/>
        <circle cx="37" cy="37" r="4" fill="#333"/>
        <circle cx="63" cy="37" r="4" fill="#333"/>
        <circle cx="37" cy="63" r="4" fill="#333"/>
        <circle cx="63" cy="63" r="4" fill="#333"/>
        <circle cx="50" cy="50" r="4" fill="#333"/>
    </svg>`;
}
