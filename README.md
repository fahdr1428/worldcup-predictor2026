# ⚽ World Cup 2026 Predictor

A Python ML pipeline + Streamlit app that predicts the 2026 FIFA World Cup using **historical match data** (~45,000 international matches since 1872) **and current player ratings** for the real 48 qualified nations (~576 players, ratings reflecting 2025/26 club form).

## What's in this version

- **5-tab Streamlit app**: Match Predictor · Team Explorer · Squad Ratings · Players & Overrides · Tournament Simulator
- **Real 2026 World Cup teams + official draw**: All 48 qualified nations in their actual groups from the December 2025 FIFA draw
- **Squad-aware predictions**: every model output factors in current player ratings, injuries, and suspensions
- **2025/26 season form baked in**: ratings reflect this season's club performance — Yamal 90, Vitinha 88, Marmoush 84, etc.
- **Override UI**: edit any player's rating with a delta (+5 for hot form, -10 for poor form), mark injuries/suspensions, changes flow instantly into predictions
- **Position-aware strength**: separate attack / midfield / defence / GK ratings so the model knows *where* a team is strong
- **Optimised Monte Carlo simulator**: batched matchup precomputation runs 5,000 simulations in ~17s locally, ~30-60s on Streamlit Cloud free tier
- **Instant-load default**: the standard simulation result is pre-computed and ships with the repo, so the app loads it in <20ms

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip3 install -r requirements.txt --break-system-packages

# 2. Build everything end-to-end (~2 minutes)
python3 -m scripts.run_pipeline

# 3. Launch the app
streamlit run app/streamlit_app.py
```

## How predictions work

For any given match (e.g. Argentina vs France in a Round of 16):

1. **ELO** — long-running team-strength rating, tournament-weighted (World Cup matches matter more than friendlies)
2. **Form** — rolling 10-game scoring/conceding rates for each team
3. **Head-to-head** — last 5 meetings between the two sides
4. **Context** — neutral venue, host country, friendly vs. tournament
5. **Squad strength** — current top-XI rating with 2025/26 form, attack/defence/midfield/GK breakdown, count of unavailable players

These ~30 features feed an XGBoost classifier predicting W/D/L, and a pair of Poisson regressors predicting goals for each side. The Poisson output gives a full 9×9 score-line probability matrix.

For the tournament, 5,000 Monte Carlo trials walk every team through groups → knockouts and tally how often each reaches each round.

## The official 2026 groups (December 2025 draw)

```
A: Mexico, South Korea, South Africa, Czech Republic
B: Canada, Switzerland, Qatar, Bosnia and Herzegovina
C: Brazil, Morocco, Haiti, Scotland
D: United States, Paraguay, Australia, Turkey
E: Germany, Curacao, Ivory Coast, Ecuador
F: Netherlands, Japan, Sweden, Tunisia
G: Belgium, Egypt, Iran, New Zealand
H: Spain, Cape Verde, Saudi Arabia, Uruguay
I: France, Senegal, Iraq, Norway
J: Argentina, Algeria, Austria, Jordan
K: Portugal, DR Congo, Uzbekistan, Colombia
L: England, Croatia, Ghana, Panama
```

## Honest limitations

- **Squad data is current, not historical**: When the model trains on a 2002 Brazil vs Argentina match, it uses *today's* squad strength as a proxy. Historical squad-strength time-series don't exist publicly. This is fine because (a) the model mainly trains on 1990+ matches where the dominant signal is ELO and form, (b) squad strength is most valuable for the *2026* matches we're actually predicting.
- **Player ratings are hand-curated approximations**: They reflect 2025/26 club form, drawing on FIFA-game ratings and recent international form, but aren't pulled live from any single source. Use the in-app Override UI to bump ratings as the tournament approaches.
- **Penalty shootouts**: Knockout draws are split 50/50 between the two teams (fine approximation).

## Tech stack

XGBoost · scikit-learn · pandas · NumPy · SciPy (Poisson) · BeautifulSoup4 (scraper) · Streamlit · Altair · joblib · tqdm

## License

Data from [martj42/international_results](https://github.com/martj42/international_results) (MIT). Player ratings are inspired by FIFA-game data, included here in a derived form. Predictor code MIT-licensed.
