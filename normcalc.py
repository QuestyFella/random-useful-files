import math


def pdf(x, mu=0, sigma=1):
    """Probability density function."""
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def cdf(x, mu=0, sigma=1):
    """Cumulative distribution function P(X <= x)."""
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def _erfinv(z):
    """Inverse error function via Newton's method."""
    if not -1 < z < 1:
        raise ValueError("z must be in (-1, 1)")
    x = 0.0
    for _ in range(100):
        fx = math.erf(x) - z
        dfx = (2 / math.sqrt(math.pi)) * math.exp(-(x ** 2))
        step = fx / dfx
        x -= step
        if abs(step) < 1e-15:
            break
    return x


def ppf(p, mu=0, sigma=1):
    """Percent point function (inverse CDF) — find x given probability p."""
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1 (exclusive)")
    return mu + sigma * math.sqrt(2) * _erfinv(2 * p - 1)


def z_score(x, mu, sigma):
    """Standardize x to a z-score."""
    return (x - mu) / sigma


def prob_between(a, b, mu=0, sigma=1):
    """Probability that X falls between a and b."""
    return cdf(b, mu, sigma) - cdf(a, mu, sigma)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normal distribution calculator")
    parser.add_argument("--mu", type=float, default=0, help="Mean (default: 0)")
    parser.add_argument("--sigma", type=float, default=1, help="Std deviation (default: 1)")
    parser.add_argument("--pdf", type=float, metavar="X", help="PDF at x")
    parser.add_argument("--cdf", type=float, metavar="X", help="CDF at x, i.e. P(X <= x)")
    parser.add_argument("--ppf", type=float, metavar="P", help="Inverse CDF: x for probability p")
    parser.add_argument("--zscore", type=float, metavar="X", help="Z-score of x")
    parser.add_argument("--between", type=float, nargs=2, metavar=("A", "B"), help="P(A < X < B)")
    args = parser.parse_args()

    mu, sigma = args.mu, args.sigma
    any_calc = any([args.pdf is not None, args.cdf is not None, args.ppf is not None,
                    args.zscore is not None, args.between is not None])

    print(f"Normal Distribution (mu={mu}, sigma={sigma})")
    print("-" * 40)

    if not any_calc:
        # Default summary when no specific calculation is requested
        print(f"PDF at x=0:          {pdf(0, mu, sigma):.6f}")
        print(f"CDF at x=0:          {cdf(0, mu, sigma):.6f}")
        print(f"CDF at x=1.96:       {cdf(1.96, mu, sigma):.6f}")
        print(f"P(-1 < X < 1):       {prob_between(-1, 1, mu, sigma):.6f}")
        print(f"P(-2 < X < 2):       {prob_between(-2, 2, mu, sigma):.6f}")
        print(f"P(-3 < X < 3):       {prob_between(-3, 3, mu, sigma):.6f}")
        print(f"PPF at p=0.975:      {ppf(0.975, mu, sigma):.6f}")
        print(f"Z-score (x=0):       {z_score(0, mu, sigma):.6f}")
    else:
        if args.pdf is not None:
            print(f"PDF at x={args.pdf}:      {pdf(args.pdf, mu, sigma):.6f}")
        if args.cdf is not None:
            print(f"CDF at x={args.cdf}:      {cdf(args.cdf, mu, sigma):.6f}")
        if args.ppf is not None:
            print(f"PPF at p={args.ppf}:      {ppf(args.ppf, mu, sigma):.6f}")
        if args.zscore is not None:
            print(f"Z-score (x={args.zscore}): {z_score(args.zscore, mu, sigma):.6f}")
        if args.between is not None:
            a, b = args.between
            print(f"P({a} < X < {b}):   {prob_between(a, b, mu, sigma):.6f}")
