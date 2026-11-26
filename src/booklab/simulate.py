"""Simulates the sales data"""

import numpy as np
import pandas as pd

GENRE_ELASTICITY = {"fiction": -1.2, "thriller": -1.8, "children": -0.8, "nonfiction": -1.0}

def simulate_sales(n_titles=300, n_weeks=104, theta=0.4, sigma=0.25, seed=42):
    rng = np.random.default_rng(seed)            # reproducible randomness
    genres = rng.choice(list(GENRE_ELASTICITY), size=n_titles)
    alpha = rng.normal(3.0, 0.7, size=n_titles)  # baseline popularity
    base_price = rng.uniform(6, 20, size=n_titles).round(2)
    launch = rng.integers(-40, 60, size=n_titles)  # launch week

    rows = []
    for i in range(n_titles):
        for t in range(n_weeks):
            weeks_since = max(t - launch[i], 0)
            if t < launch[i]:
                continue                          # not on sale yet
            m = rng.binomial(1, 0.25)              # marketing push this week
            p_promo = 0.10 + 0.30 * m              # confounding: promos follow marketing
            d = rng.choice([0.0, 0.1, 0.2, 0.3]) if rng.random() < p_promo else 0.0
            price = base_price[i] * (1 - d)
            eps = GENRE_ELASTICITY[genres[i]]
            ln_q = (alpha[i] + eps * np.log(price)
                    + 0.3 * np.sin(2 * np.pi * t / 52)
                    - 0.02 * weeks_since + theta * m
                    + rng.normal(0, sigma))
            rows.append((i, genres[i], t, price, d, m, np.exp(ln_q)))
    df = pd.DataFrame(rows, columns=["title", "genre", "week", "price", "discount", "marketing", "units"])
    df["units"] = df["units"].round().clip(lower=1)   # integer sales, no zeros so logs are defined
    return df