"""arima_fit(method='css') refuses a seasonal spec whose s*P presample eats the
series with 'Lower p, d, or q, or supply a longer series' -- the remedy names
p, d, q (here p=1, d=0, q=0) and not seasonal, P or s, which are the cause."""
import numpy as np, tsecon
y = np.random.default_rng(0).standard_normal(200)
for seas in [(1, 0, 0, 199), (50, 0, 0, 4)]:
    try:
        tsecon.arima_fit(y, p=1, seasonal=seas, method="css")
    except ValueError as e:
        print(seas, "->", e, "| names seasonal:", "seasonal" in str(e))
