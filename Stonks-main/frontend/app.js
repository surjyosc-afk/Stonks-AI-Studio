// ==========================================
// STONKS AI - API Integration & UI Logic
// ==========================================

const API_BASE = 'http://localhost:8000'; 
let chartInstance = null;
let isSignalReady = false;

// Utility to format numbers into Indian Rupees
const formatRs = (num) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(num);

// --- Main API Fetch Function (Sequential to bypass Rate Limits) ---
async function analyzeStock() {
    const tickerInputEl = document.getElementById('tickerInput');
    if (!tickerInputEl) return;
    
    const ticker = tickerInputEl.value.trim().toUpperCase();
    if (!ticker) {
        alert("Please enter a valid ticker (e.g., RELIANCE.BSE or TCS.BSE)");
        return;
    }
    
    // 1. Toggle Loading State
    document.getElementById('dashboardContent').classList.add('hidden');
    document.getElementById('loadingState').classList.remove('hidden');
    document.getElementById('loadingState').classList.add('flex');
    document.getElementById('displayTicker').innerText = ticker;

    // Reset the Reveal Button
    isSignalReady = false;
    document.getElementById('signalResultData').classList.add('hidden');
    document.getElementById('signalResultData').classList.remove('flex');
    
    const revealBtn = document.getElementById('revealSignalBtn');
    revealBtn.classList.remove('hidden');
    revealBtn.classList.add('opacity-50', 'cursor-not-allowed', 'from-purple-600/80', 'to-blue-600/80');
    revealBtn.classList.remove('from-green-600/80', 'to-emerald-600/80', 'from-red-600/80', 'to-orange-600/80', 'from-gray-500/80', 'to-gray-400/80');
    revealBtn.disabled = true;
    document.getElementById('btnIcon').className = 'fas fa-spinner fa-spin text-white';
    document.getElementById('btnText').innerText = 'Analyzing Real Market Data...';

    try {
        // 2. Fetch Prediction FIRST to trigger the Python cache
        const predRes = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker, period: "6mo", use_mock: false }) 
        });

        if (!predRes.ok) {
            const errData = await predRes.json();
            throw new Error(errData.detail || "Prediction API failed.");
        }
        const predData = await predRes.json();

        // 3. Fetch Signals & Patterns SEQUENTIALLY (These hit the Python cache instantly)
        const sigRes = await fetch(`${API_BASE}/signals/${ticker}?period=6mo&use_mock=false&include_news=true`);
        if (!sigRes.ok) throw new Error("Signals API failed.");
        const sigData = await sigRes.json();

        const patRes = await fetch(`${API_BASE}/patterns/${ticker}?period=1y&use_mock=false`);
        if (!patRes.ok) throw new Error("Patterns API failed.");
        const patData = await patRes.json();

        // 4. Send the data to the UI function
        populateDashboard(ticker, predData, sigData, patData);

    } catch (error) {
        console.error("API Error:", error);
        alert(`Connection failed: ${error.message}\n\nEnsure your Python backend is running and you are using a valid suffix (e.g. .BSE).`);
    } finally {
        // Hide loader, show dashboard
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('loadingState').classList.remove('flex');
        document.getElementById('dashboardContent').classList.remove('hidden');
    }
}

// --- UI Rendering Function ---
function populateDashboard(ticker, predData, sigData, patData) {
    
    // 1. Update Price & Prediction
    const currPrice = predData.current_price;
    const predPrice = predData.predicted_price;
    document.getElementById('currentPrice').innerText = formatRs(currPrice);
    
    const priceEl = document.getElementById('predictedPrice');
    const isUp = predPrice >= currPrice;
    priceEl.innerHTML = `${formatRs(predPrice)} <i class="fas fa-arrow-${isUp ? 'up' : 'down'} text-xl ml-2 ${isUp ? 'text-green-400' : 'text-red-400'}"></i>`;
    priceEl.className = `text-4xl font-bold flex items-center gap-2 text-white`;

    // 2. Prepare Radar/Signals Badge
    const badgeContainer = document.getElementById('alertBadgeContainer');
    const badge = document.getElementById('alertBadge');
    badge.innerText = `COMPOSITE: ${sigData.alert_type}`;
    
    const revealBtn = document.getElementById('revealSignalBtn');
    revealBtn.classList.remove('from-purple-600/80', 'to-blue-600/80');
    
    if (sigData.alert_type === 'BUY') {
        badgeContainer.className = 'p-[1px] rounded-full bg-gradient-to-r from-green-500 to-emerald-700 mb-4 shadow-[0_0_15px_rgba(34,197,94,0.3)] w-full max-w-[200px] mx-auto';
        badge.className = 'block px-6 py-1.5 rounded-full text-sm font-bold bg-[#0b0c10] text-green-400 tracking-wider';
        revealBtn.classList.add('from-green-600/80', 'to-emerald-600/80');
        document.getElementById('btnIcon').className = 'fas fa-unlock text-green-300';
    } else if (sigData.alert_type === 'SELL') {
        badgeContainer.className = 'p-[1px] rounded-full bg-gradient-to-r from-red-500 to-rose-700 mb-4 shadow-[0_0_15px_rgba(239,68,68,0.3)] w-full max-w-[200px] mx-auto';
        badge.className = 'block px-6 py-1.5 rounded-full text-sm font-bold bg-[#0b0c10] text-red-400 tracking-wider';
        revealBtn.classList.add('from-red-600/80', 'to-orange-600/80');
        document.getElementById('btnIcon').className = 'fas fa-unlock text-red-300';
    } else {
        badgeContainer.className = 'p-[1px] rounded-full bg-gradient-to-r from-gray-600 to-gray-500 mb-4 w-full max-w-[200px] mx-auto';
        badge.className = 'block px-6 py-1.5 rounded-full text-sm font-bold bg-[#0b0c10] text-gray-300 tracking-wider';
        revealBtn.classList.add('from-gray-500/80', 'to-gray-400/80');
        document.getElementById('btnIcon').className = 'fas fa-unlock text-gray-300';
    }
    
    document.getElementById('radarScore').innerText = (sigData.score > 0 ? "+" : "") + sigData.score.toFixed(2);
    
    let detailsHtml = `<ul class="list-disc pl-4 space-y-1.5">`;
    if (sigData.signals && sigData.signals.length > 0) {
        sigData.signals.forEach(sig => {
            detailsHtml += `<li>${sig}</li>`;
        });
    } else {
        detailsHtml += `<li>No major signals detected.</li>`;
    }
    detailsHtml += `</ul>`;
    document.getElementById('radarDetails').innerHTML = detailsHtml;

    // Activate the Reveal Button
    isSignalReady = true;
    revealBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    revealBtn.disabled = false;
    document.getElementById('btnText').innerText = 'Signal Locked. Click to Reveal.';

    // 3. Update FinBERT Sentiment
    const sentimentScore = (sigData.components && sigData.components.sentiment) ? sigData.components.sentiment : 0; 
    const sentLabelEl = document.getElementById('sentimentLabel');
    const sentIconEl = document.getElementById('sentimentIcon');
    const sentBarEl = document.getElementById('sentimentBar');

    let widthPct = ((sentimentScore + 1) / 2) * 100;
    widthPct = Math.max(5, Math.min(95, widthPct)); 
    sentBarEl.style.width = `${widthPct}%`;

    if (sentimentScore > 0.2) {
        sentLabelEl.innerText = "Bullish";
        sentIconEl.className = "fas fa-arrow-trend-up text-2xl text-green-400 drop-shadow-[0_0_8px_rgba(34,197,94,0.8)]";
        sentBarEl.className = "bg-green-500 h-2 rounded-full transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(34,197,94,0.8)]";
    } else if (sentimentScore < -0.2) {
        sentLabelEl.innerText = "Bearish";
        sentIconEl.className = "fas fa-arrow-trend-down text-2xl text-red-400 drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]";
        sentBarEl.className = "bg-red-500 h-2 rounded-full transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(239,68,68,0.8)]";
    } else {
        sentLabelEl.innerText = "Neutral";
        sentIconEl.className = "fas fa-minus text-2xl text-gray-400";
        sentBarEl.className = "bg-gray-500 h-2 rounded-full transition-all duration-1000 ease-out";
    }

    // 4. Update Technical Patterns
    document.getElementById('supLevel').innerText = patData.support && patData.support.length > 0 ? formatRs(patData.support[patData.support.length - 1]) : "--";
    document.getElementById('resLevel').innerText = patData.resistance && patData.resistance.length > 0 ? formatRs(patData.resistance[patData.resistance.length - 1]) : "--";
    
    const patList = document.getElementById('patternList');
    patList.innerHTML = ''; 
    
    const allPatterns = [
        ...(patData.breakouts || []).map(p => ({name: "Breakout Vector", desc: p.type || "Detected", icon: "fa-bolt", color: "text-blue-400"})),
        ...(patData.trend_reversals || []).map(p => ({name: "Reversal Matrix", desc: p.type || "Detected", icon: "fa-retweet", color: "text-purple-400"})),
        ...(patData.head_and_shoulders || []).map(p => ({name: "H&S Formation", desc: "Topology detected", icon: "fa-mountain", color: "text-indigo-400"}))
    ];

    if (allPatterns.length === 0) {
        patList.innerHTML = '<li class="text-xs text-gray-500 italic px-2 py-4 text-center">No anomalies detected in spatial window.</li>';
    } else {
        allPatterns.forEach(p => {
            patList.innerHTML += `
                <li class="flex items-center gap-4 p-3 bg-[#0b0c10]/60 rounded-xl border border-white/5 hover:bg-white/5 transition-colors backdrop-blur-sm">
                    <div class="h-8 w-8 rounded-full bg-[#12141a] border border-white/10 flex items-center justify-center shadow-inner">
                        <i class="fas ${p.icon} text-sm ${p.color}"></i>
                    </div>
                    <div>
                        <p class="text-sm font-bold text-white tracking-wide">${p.name}</p>
                        <p class="text-[10px] text-gray-400 uppercase tracking-widest">${p.desc.replace('_', ' ')}</p>
                    </div>
                </li>
            `;
        });
    }

    // Render Pure Data Chart using historical arrays
    renderChart(
        predData.historical_dates || [], 
        predData.historical_prices || [currPrice], 
        predPrice
    );
}

// --- Toggle Button Logic ---
function toggleSignal() {
    if (!isSignalReady) return;
    document.getElementById('revealSignalBtn').classList.add('hidden');
    const signalData = document.getElementById('signalResultData');
    signalData.classList.remove('hidden');
    signalData.classList.add('flex');
}

// --- Real Data Chart ---
function renderChart(historicalDates, historicalPrices, predictedPrice) {
    const ctx = document.getElementById('stockChart').getContext('2d');
    if (chartInstance) chartInstance.destroy();

    // Combine history with the new future prediction
    const labels = [...historicalDates, "T+1 (Target)"];
    const dataPts = [...historicalPrices, predictedPrice];

    let gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(139, 92, 246, 0.5)'); 
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)'); 

    // Find the current price (last item in history) to determine prediction color
    const currentPrice = historicalPrices[historicalPrices.length - 1];
    const isBullish = predictedPrice >= currentPrice;

    // Make only the final prediction point visible
    const pointRadii = dataPts.map((_, index) => index === dataPts.length - 1 ? 8 : 0);
    const pointColors = dataPts.map((_, index) => index === dataPts.length - 1 ? (isBullish ? '#4ade80' : '#f87171') : 'transparent');

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Price Vector',
                data: dataPts,
                borderColor: '#8b5cf6',
                backgroundColor: gradient,
                borderWidth: 3,
                pointBackgroundColor: pointColors,
                pointBorderColor: pointColors,
                pointRadius: pointRadii,
                pointHoverRadius: 10,
                fill: true,
                tension: 0.4 // THIS IS THE MAGIC PROPERTY THAT CURVES THE LINE!
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { display: false },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                y: { display: true, grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false }, ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 }, callback: function(value) { return '₹' + value; } } },
                x: { grid: { display: false, drawBorder: false }, ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 }, maxTicksLimit: 6 } }
            },
            interaction: { intersect: false, mode: 'index' },
        }
    });
}