"""Policy factory: generate a large, diverse corpus of CORRECT policy -> JSON pairs.

The structured policy is sampled first, the canonical JSON is derived from it, and the
natural-language side is rendered with varied phrasing. Language is configurable:

  - hinglish : romanized Hindi mixed with English, the way Indians actually write
               payment instructions ("Mujhe har mahine groceries ke liye 5 hazaar
               tak kharch karne do, kahin bhi, kabhi bhi.")
  - english  : plain English
  - mixed    : per-pair random choice of the two

Phrasing deliberately includes non-literal-but-equivalent forms that are HARD to get
right: "kabhi bhi"/"din bhar" == 00:00-23:59, "kahin bhi" == any merchant,
"5 hazaar"/"1 lakh"/"₹5,000" == the integer amount. Such pairs are flagged
equivalence_hard so the harness can measure false-reject rate on them separately.

Output row: {"english", "json", "equivalence_hard", "lang", "policy_id"}
(the field stays named "english" so the rest of the harness is unchanged; it holds the
Hinglish text when lang == "hinglish".)

Run standalone:
  python generate_pairs.py --n 2400 --language hinglish --out data/policies_faithful.jsonl
"""
import argparse
import random

from utils import write_jsonl

# category -> realistic Indian merchant pool (empty => usually "any")
CATEGORY_MERCHANTS = {
    "groceries": ["BigBasket", "DMart", "Reliance Fresh", "Blinkit", "Zepto"],
    "fuel": ["HPCL", "IndianOil", "BharatPetroleum", "Shell"],
    "food_delivery": ["Swiggy", "Zomato"],
    "dining": [],
    "pharmacy": ["Apollo", "MedPlus", "Netmeds", "PharmEasy", "Tata1mg"],
    "travel": ["MakeMyTrip", "IRCTC", "Cleartrip", "Goibibo"],
    "hotels": ["OYO", "Taj", "Marriott", "Treebo"],
    "online_shopping": ["Amazon", "Flipkart", "Myntra", "Meesho"],
    "ride_hailing": ["Uber", "Ola", "Rapido"],
    "entertainment": ["BookMyShow", "PVR", "INOX"],
    "utilities": [],
    "education": ["Byjus", "Unacademy"],
    "rent": ["NoBroker", "NestAway"],
    "streaming": ["Netflix", "Hotstar", "JioCinema"],
    "mobile_recharge": ["Jio", "Airtel", "Vi"],
    "electricity_bill": [],
    "gold": ["Tanishq", "Kalyan", "CaratLane"],
    "metro": ["DMRC", "Namma Metro"],
}
CATEGORIES = list(CATEGORY_MERCHANTS)
CURRENCIES = ["INR", "INR", "INR", "INR", "USD"]   # INR-heavy
PERIODS = ["per_transaction", "daily", "weekly", "monthly"]
AMOUNT_RANGE = {
    "per_transaction": (200, 5000), "daily": (500, 10000),
    "weekly": (2000, 30000), "monthly": (5000, 100000),
}
WINDOWS = [
    ("06:00", "22:00"), ("09:00", "18:00"), ("08:00", "21:00"), ("17:00", "23:00"),
    ("07:00", "12:00"), ("10:00", "20:00"), ("05:00", "23:00"), ("06:00", "23:59"),
    ("00:00", "23:59"),  # all-day
]


# ----------------------------------------------------------- structured sample
def sample_policy(rng):
    period = rng.choice(PERIODS)
    cur = rng.choice(CURRENCIES)
    lo, hi = AMOUNT_RANGE[period]
    step = 100 if hi <= 10000 else 500
    amt = rng.randrange(lo, hi + 1, step)
    n_cat = rng.choices([1, 2, 3], weights=[5, 3, 1])[0]
    cats = rng.sample(CATEGORIES, n_cat)
    window = rng.choice(WINDOWS)
    pools = [CATEGORY_MERCHANTS[c] for c in cats]
    if all(pools) and rng.random() < 0.6:
        pool = pools[0]
        merchants = sorted(rng.sample(pool, min(len(pool), rng.choice([1, 2, 2]))))
    else:
        merchants = "any"
    return {"max_amount": amt, "currency": cur, "period": period,
            "categories": cats, "merchants": merchants,
            "time_window": {"start": window[0], "end": window[1]}}


def _join(rng, items, conjs):
    items = [i.replace("_", " ") for i in items]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + rng.choice(conjs) + items[-1]


# --------------------------------------------------------------- ENGLISH render
PERIOD_PHRASE_EN = {
    "per_transaction": ["per transaction", "for each transaction", "on a single transaction"],
    "daily": ["per day", "a day", "each day", "daily"],
    "weekly": ["per week", "a week", "weekly"],
    "monthly": ["per month", "a month", "monthly"],
}
LEAD_EN = ["Allow payments up to", "Permit up to", "Approve up to",
           "Let me spend up to", "Authorize up to"]


def _to12h(hhmm):
    h, m = [int(x) for x in hhmm.split(":")]
    suf = "am" if h < 12 else "pm"
    return f"{h % 12 or 12}{'' if m == 0 else ':%02d' % m}{suf}"


def _amt_en(rng, amt, cur):
    if cur == "INR":
        return rng.choice([f"{amt} rupees", f"₹{amt}", f"Rs. {amt}", f"₹{amt:,}"]), False
    if cur == "USD":
        return rng.choice([f"${amt}", f"{amt} dollars"]), False
    return rng.choice([f"€{amt}", f"{amt} euros"]), False


def render_english(rng, p):
    amt_p, eh_a = _amt_en(rng, p["max_amount"], p["currency"])
    per_p = rng.choice(PERIOD_PHRASE_EN[p["period"]])
    cat_p = _join(rng, p["categories"], [" and ", " or "])
    if p["merchants"] == "any":
        mer_p = rng.choice([" at any merchant", " anywhere", " at any store", ""]); eh_m = True
    else:
        mer_p = rng.choice([" at ", " only at "]) + _join(rng, p["merchants"], [" and ", " or "]); eh_m = False
    w = (p["time_window"]["start"], p["time_window"]["end"])
    if w == ("00:00", "23:59"):
        tim_p = rng.choice([", any time", " at any time of day", " around the clock"]); eh_t = True
    else:
        tim_p = rng.choice([f", between {_to12h(w[0])} and {_to12h(w[1])}",
                            f", from {w[0]} to {w[1]}",
                            f" during {_to12h(w[0])}-{_to12h(w[1])}"]); eh_t = False
    if rng.random() < 0.6:
        text = f"{rng.choice(LEAD_EN)} {amt_p} {per_p} for {cat_p}{mer_p}{tim_p}."
    else:
        text = f"Cap {p['period'].replace('_','-')} spending on {cat_p} at {amt_p}{mer_p}{tim_p}."
    return text, bool(eh_a or eh_m or eh_t)


# -------------------------------------------------------------- HINGLISH render
PERIOD_PHRASE_HI = {
    "per_transaction": ["ek transaction mein", "har transaction pe", "ek baar mein"],
    "daily": ["har din", "roz", "daily", "din ka"],
    "weekly": ["har hafte", "weekly", "hafte ka"],
    "monthly": ["har mahine", "monthly", "mahine ka"],
}


def _hin_time(hhmm):
    h = int(hhmm.split(":")[0])
    if 4 <= h <= 10:
        tod = "subah"
    elif 11 <= h <= 15:
        tod = "dopahar"
    elif 16 <= h <= 18:
        tod = "shaam"
    else:
        tod = "raat"
    return f"{tod} {h % 12 or 12} baje"


def _amt_hi(rng, amt, cur):
    if cur == "INR":
        opts = [f"{amt} rupaye", f"₹{amt}", f"{amt} rs", f"₹{amt:,}"]
        eh = False
        if amt % 100000 == 0:
            opts += [f"{amt // 100000} lakh rupaye", f"{amt // 100000} lakh"]; eh = True
        elif amt % 1000 == 0:
            opts += [f"{amt // 1000} hazaar rupaye", f"{amt // 1000} hazaar"]; eh = True
        choice = rng.choice(opts)
        return choice, (eh and ("hazaar" in choice or "lakh" in choice))
    if cur == "USD":
        return rng.choice([f"${amt}", f"{amt} dollar"]), False
    return rng.choice([f"€{amt}", f"{amt} euro"]), False


def render_hinglish(rng, p):
    amt_p, eh_a = _amt_hi(rng, p["max_amount"], p["currency"])
    per_p = rng.choice(PERIOD_PHRASE_HI[p["period"]])
    cat_p = _join(rng, p["categories"], [" aur ", " ya "])
    if p["merchants"] == "any":
        mer_p = rng.choice([" kisi bhi merchant pe", " kahin bhi", " kisi bhi store pe", ""]); eh_m = True
    else:
        lead = rng.choice(["sirf ", "", "bas "])
        mer_p = " " + lead + _join(rng, p["merchants"], [" aur ", " ya "]) + " pe"; eh_m = False
    w = (p["time_window"]["start"], p["time_window"]["end"])
    if w == ("00:00", "23:59"):
        tim_p = rng.choice([", kabhi bhi", " din bhar", ", 24 ghante", ", kisi bhi time"]); eh_t = True
    else:
        tim_p = f", {_hin_time(w[0])} se {_hin_time(w[1])} tak"; eh_t = False

    cat_cap = cat_p[0].upper() + cat_p[1:]
    text = rng.choice([
        f"Mujhe {per_p} {cat_p} ke liye {amt_p} tak kharch karne do{mer_p}{tim_p}.",
        f"{cat_cap} pe {per_p} {amt_p} tak allow karo{mer_p}{tim_p}.",
        f"{amt_p} tak {cat_p} ke liye {per_p} approve karo{mer_p}{tim_p}.",
        f"{per_p.capitalize()} {cat_p} ka {amt_p} tak ka limit rakho{mer_p}{tim_p}.",
    ])
    return text, bool(eh_a or eh_m or eh_t)


RENDERERS = {"hinglish": render_hinglish, "english": render_english}


def make_pair(rng, language):
    lang = rng.choice(["hinglish", "english"]) if language == "mixed" else language
    p = sample_policy(rng)
    text, eh = RENDERERS[lang](rng, p)
    return {"english": text, "json": p, "equivalence_hard": eh, "lang": lang}


def generate(n, seed, language="hinglish"):
    rng = random.Random(seed)
    seen, out, attempts = set(), [], 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        pr = make_pair(rng, language)
        if pr["english"] in seen:
            continue
        seen.add(pr["english"])
        pr["policy_id"] = f"p{len(out):05d}"
        out.append(pr)
    eh = sum(x["equivalence_hard"] for x in out)
    langs = {}
    for x in out:
        langs[x["lang"]] = langs.get(x["lang"], 0) + 1
    print(f"[generate] {len(out)} unique pairs | langs {langs} | "
          f"equivalence-hard {eh} ({eh/len(out):.0%})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--language", default="hinglish", choices=["hinglish", "english", "mixed"])
    ap.add_argument("--out", default="data/policies_faithful.jsonl")
    args = ap.parse_args()
    write_jsonl(args.out, generate(args.n, args.seed, args.language))
    print(f"[generate] wrote {args.out}")
