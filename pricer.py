import math
from scipy.stats import norm


def calc_d1_d2(S : float, K : float, T : float, r : float, sigma: float
) -> tuple[float]:

    d1 = (math.log(S/K) + (r + 0.5*(sigma**2))*T)/ (sigma* (math.sqrt(T)))
    d2 = d1 - sigma*(math.sqrt(T))

    return d1,d2


def black_scholes(
        S : float, K : float, T : float, r : float, sigma: float, option_type: str = 'call'
) -> float:
    #Calculates the Black-Scholes option price for a Call or Put.#

    (d1 , d2) = calc_d1_d2(S,K,T,r,sigma)


    if option_type.lower() == "call":
        price = S*norm.cdf(d1) - K* math.exp(-r*T)*norm.cdf(d2)

    elif option_type.lower() == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    else:
        raise ValueError("option_type must be either 'call' or 'put'. ") 

    return price


def calculate_greeks(
        S : float, K : float, T : float, r : float, sigma: float, option_type: str = 'call'
) -> dict:

    if option_type.lower() != "call" and option_type.lower()!="put":
        raise ValueError("option_type must be either 'call' or 'put'. ")


    
    (d1 , d2) = calc_d1_d2(S,K,T,r,sigma)


    pdf_d1 = norm.pdf(d1)

    #1. Delta
    if option_type.lower()=="call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1)-1


    #2. Gamma (Same for Call and Put)
    gamma = pdf_d1/(S* sigma* math.sqrt(T))


    #3. Vega (1% change in volatility)
    vega = (S* pdf_d1* math.sqrt(T))/100

    norm_cdf_d2 = norm.cdf(d2)
    norm_cdf_d2_1 = norm.cdf(-d2)


    #4. Theta (1-day decay)
    if option_type.lower()=="call":
        theta_annual = -( (S*pdf_d1*sigma)/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm_cdf_d2)
    else:
        theta_annual = -( (S*pdf_d1*sigma)/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm_cdf_d2_1)

    theta_daily = theta_annual/365.0


    # 5. Rho (1% change in interest rate)
    if option_type.lower()=="call":
        rho = (K*T*math.exp(-r*T)*norm_cdf_d2)/100.0
    else:
        rho = -(K*T*math.exp(-r*T)*norm.cdf(-d2))/100.0

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta (1D)": theta_daily,
        "Rho": rho,
    }



def calculate_unscaled_vega(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Calculates unscaled Vega dC/d_sigma for Newton-Raphson derivative step."""
    (d1 ,_) = calc_d1_d2(S,K,T,r,sigma)

    return S * norm.pdf(d1) * math.sqrt(T)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Back-calculates Implied Volatility (sigma) given an option market price using Newton-Raphson."""
    sigma = 0.25  # Initial guess (25% volatility)

    for i in range(max_iter):
        bs_price = black_scholes(S,K,T,r,sigma,option_type)
        vega = calculate_unscaled_vega(S,K,T,r,sigma)

        diff = bs_price - market_price


        # Convergence Check
        if abs(diff) < tol:
            return sigma

        # Avoid division by zero as Vega approaches Zero
        if abs(vega)< 1e-8:
            break

        sigma = sigma - diff/vega

    raise ValueError(
        f"Implied Volatility did not converge after {max_iter} iterations."
    )








if __name__ == "__main__":
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    target_sigma = 0.20  # 20% volatility

    # 1. Price option at 20% volatility
    mkt_price = black_scholes(S, K, T, r, target_sigma, "call")
    print(f"Observed Market Price : ${mkt_price:.4f}")

    # 2. Back-calculate IV from the price
    calc_iv = implied_volatility(mkt_price, S, K, T, r, "call")
    print(f"Calculated Implied Vol : {calc_iv * 100:.2f}%")