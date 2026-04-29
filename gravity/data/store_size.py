"""
Store square footage estimator.

Uses brand and category lookup tables to estimate store sizes when real
measurements are unavailable (e.g., data sourced from OSM which does not
report square footage).

Usage
-----
    from gravity.data.store_size import estimate_store_size, estimate_sqft

    stores = estimate_store_size(stores)          # mutates in-place & returns
    sqft, source = estimate_sqft("McDonald's")    # single-store lookup
"""

from __future__ import annotations

from typing import Optional

from gravity.data.schema import Store

# ---------------------------------------------------------------------------
# Brand lookup  (200+ US retail brands -> average square footage)
# ---------------------------------------------------------------------------

BRAND_SQFT: dict[str, float] = {
    # ── Grocery / Supermarket ───────────────────────────────────────────
    "Walmart Supercenter": 180_000,
    "Walmart Neighborhood Market": 42_000,
    "Walmart": 150_000,
    "Costco": 146_000,
    "Kroger": 66_000,
    "Trader Joe's": 12_000,
    "Whole Foods": 40_000,
    "Whole Foods Market": 40_000,
    "Aldi": 12_000,
    "Publix": 48_000,
    "H-E-B": 70_000,
    "HEB": 70_000,
    "Safeway": 55_000,
    "Target": 130_000,
    "Sam's Club": 136_000,
    "BJ's Wholesale Club": 115_000,
    "BJ's": 115_000,
    "Meijer": 190_000,
    "WinCo Foods": 85_000,
    "Food Lion": 35_000,
    "Giant Eagle": 55_000,
    "Giant Food": 52_000,
    "Stop & Shop": 55_000,
    "ShopRite": 55_000,
    "Wegmans": 100_000,
    "Sprouts Farmers Market": 28_000,
    "Sprouts": 28_000,
    "Harris Teeter": 50_000,
    "Albertsons": 50_000,
    "Vons": 50_000,
    "Ralphs": 50_000,
    "Fred Meyer": 150_000,
    "Piggly Wiggly": 30_000,
    "Save-A-Lot": 16_000,
    "Grocery Outlet": 18_000,
    "Natural Grocers": 15_000,
    "Fresh Market": 21_000,
    "The Fresh Market": 21_000,
    "Lucky": 42_000,
    "Stater Bros": 38_000,
    "Winn-Dixie": 42_000,
    "Raley's": 50_000,
    "Market Basket": 60_000,
    "Hannaford": 48_000,
    "Ingles": 48_000,
    "Brookshire's": 42_000,
    "Lidl": 20_000,

    # ── Fast Food ───────────────────────────────────────────────────────
    "McDonald's": 4_000,
    "McDonalds": 4_000,
    "Starbucks": 1_800,
    "Subway": 1_200,
    "Chick-fil-A": 5_000,
    "Chick Fil A": 5_000,
    "Taco Bell": 2_500,
    "Wendy's": 3_500,
    "Wendys": 3_500,
    "Burger King": 3_500,
    "Dunkin'": 2_000,
    "Dunkin Donuts": 2_000,
    "Dunkin": 2_000,
    "Popeyes": 2_400,
    "Sonic Drive-In": 1_800,
    "Sonic": 1_800,
    "Jack in the Box": 2_800,
    "Whataburger": 3_500,
    "Arby's": 2_800,
    "Arbys": 2_800,
    "Hardee's": 3_000,
    "Carl's Jr": 3_000,
    "Carl's Jr.": 3_000,
    "KFC": 3_000,
    "Pizza Hut": 2_800,
    "Domino's": 1_500,
    "Dominos": 1_500,
    "Papa John's": 1_400,
    "Papa Johns": 1_400,
    "Little Caesars": 1_200,
    "Panda Express": 2_200,
    "Five Guys": 2_500,
    "In-N-Out Burger": 2_500,
    "In-N-Out": 2_500,
    "Raising Cane's": 3_200,
    "Wingstop": 1_700,
    "Jimmy John's": 1_400,
    "Jimmy Johns": 1_400,
    "Jersey Mike's": 1_400,
    "Jersey Mikes": 1_400,
    "Firehouse Subs": 1_400,
    "Zaxby's": 3_500,
    "Zaxbys": 3_500,
    "Culver's": 4_200,
    "Culvers": 4_200,
    "Shake Shack": 3_500,
    "Cookout": 2_000,
    "Tropical Smoothie Cafe": 1_400,
    "Smoothie King": 1_200,
    "Jamba Juice": 1_200,
    "Jamba": 1_200,
    "Tim Hortons": 2_200,
    "Del Taco": 2_200,
    "Church's Chicken": 2_200,
    "Checkers": 1_200,
    "Rally's": 1_200,
    "White Castle": 2_000,
    "Krispy Kreme": 3_000,
    "Wawa": 5_500,
    "Sheetz": 5_500,
    "QuikTrip": 5_000,
    "QT": 5_000,

    # ── Casual Dining / Restaurants ─────────────────────────────────────
    "Olive Garden": 8_000,
    "Applebee's": 6_000,
    "Applebees": 6_000,
    "Chili's": 6_500,
    "Chilis": 6_500,
    "Denny's": 4_500,
    "Dennys": 4_500,
    "IHOP": 4_500,
    "Panera Bread": 4_500,
    "Panera": 4_500,
    "Chipotle": 2_500,
    "Chipotle Mexican Grill": 2_500,
    "Red Lobster": 8_000,
    "TGI Friday's": 6_000,
    "TGI Fridays": 6_000,
    "Outback Steakhouse": 6_400,
    "Outback": 6_400,
    "Texas Roadhouse": 7_000,
    "Cracker Barrel": 10_000,
    "Buffalo Wild Wings": 6_000,
    "Red Robin": 5_500,
    "LongHorn Steakhouse": 6_500,
    "Waffle House": 1_600,
    "Bob Evans": 5_500,
    "Perkins": 5_500,
    "Golden Corral": 10_500,
    "Cheddar's": 8_000,
    "Cheddar's Scratch Kitchen": 8_000,
    "Cheesecake Factory": 10_000,
    "The Cheesecake Factory": 10_000,
    "P.F. Chang's": 6_500,
    "PF Changs": 6_500,
    "Hooters": 5_500,
    "BJ's Restaurant": 8_000,
    "Yard House": 8_500,
    "Benihana": 6_000,
    "Noodles & Company": 2_800,
    "Qdoba": 2_400,
    "Moe's Southwest Grill": 2_400,
    "Moes": 2_400,
    "McAlister's Deli": 3_500,
    "Jason's Deli": 3_800,
    "Potbelly": 2_000,
    "Sweetgreen": 2_200,
    "Cava": 2_400,
    "Wingstop": 1_700,

    # ── Coffee / Cafe ───────────────────────────────────────────────────
    "Peet's Coffee": 1_800,
    "Peets Coffee": 1_800,
    "Caribou Coffee": 1_800,
    "Dutch Bros": 900,
    "Dutch Bros Coffee": 900,
    "Scooter's Coffee": 800,

    # ── Apparel ─────────────────────────────────────────────────────────
    "Gap": 10_000,
    "Old Navy": 15_000,
    "H&M": 18_000,
    "Zara": 15_000,
    "TJ Maxx": 25_000,
    "TJMaxx": 25_000,
    "Ross": 22_000,
    "Ross Dress for Less": 22_000,
    "Marshalls": 25_000,
    "Burlington": 50_000,
    "Burlington Coat Factory": 50_000,
    "Forever 21": 12_000,
    "American Eagle": 6_000,
    "American Eagle Outfitters": 6_000,
    "Abercrombie & Fitch": 8_000,
    "Hollister": 5_000,
    "Banana Republic": 7_000,
    "Express": 6_000,
    "Nike": 10_000,
    "Nike Factory": 12_000,
    "Adidas": 8_000,
    "Under Armour": 8_000,
    "Lululemon": 3_500,
    "Athleta": 4_500,
    "J.Crew": 8_000,
    "J Crew": 8_000,
    "Ann Taylor": 5_000,
    "LOFT": 5_000,
    "Torrid": 4_000,
    "Lane Bryant": 5_000,
    "Chico's": 3_500,
    "Talbots": 4_000,
    "White House Black Market": 3_000,
    "Uniqlo": 10_000,
    "Hot Topic": 1_800,
    "Buckle": 4_500,
    "Foot Locker": 3_500,
    "Finish Line": 5_500,
    "DSW": 18_000,
    "Famous Footwear": 8_000,
    "Skechers": 6_000,
    "New Balance": 4_000,
    "Nordstrom Rack": 33_000,

    # ── Electronics ─────────────────────────────────────────────────────
    "Best Buy": 45_000,
    "Apple Store": 5_000,
    "Apple": 5_000,
    "GameStop": 1_500,
    "Micro Center": 55_000,
    "T-Mobile": 2_000,
    "Verizon": 3_000,
    "AT&T": 3_000,
    "Sprint": 2_500,
    "Samsung": 4_000,

    # ── Home / Hardware ─────────────────────────────────────────────────
    "Home Depot": 105_000,
    "The Home Depot": 105_000,
    "Lowe's": 112_000,
    "Lowes": 112_000,
    "Ace Hardware": 12_000,
    "IKEA": 300_000,
    "Bed Bath & Beyond": 45_000,
    "Bed Bath": 45_000,
    "Pottery Barn": 10_000,
    "Williams-Sonoma": 6_000,
    "Crate & Barrel": 15_000,
    "West Elm": 8_000,
    "Pier 1": 9_000,
    "Restoration Hardware": 20_000,
    "RH": 20_000,
    "World Market": 15_000,
    "Cost Plus": 15_000,
    "HomeGoods": 22_000,
    "At Home": 70_000,
    "Rooms To Go": 50_000,
    "Ashley Furniture": 35_000,
    "La-Z-Boy": 8_000,
    "Ethan Allen": 12_000,
    "Mattress Firm": 5_000,
    "Sleep Number": 2_000,
    "Harbor Freight": 15_000,
    "Tractor Supply": 19_000,
    "Tractor Supply Co": 19_000,
    "True Value": 10_000,
    "Menards": 160_000,
    "Floor & Decor": 75_000,
    "Lumber Liquidators": 5_500,
    "Sherwin-Williams": 3_500,
    "Benjamin Moore": 2_500,

    # ── Pharmacy / Drugstore ────────────────────────────────────────────
    "CVS": 10_000,
    "CVS Pharmacy": 10_000,
    "Walgreens": 14_000,
    "Rite Aid": 11_000,

    # ── Dollar / Discount ───────────────────────────────────────────────
    "Dollar General": 7_400,
    "Dollar Tree": 9_000,
    "Family Dollar": 8_000,
    "Five Below": 8_500,
    "99 Cents Only": 9_000,
    "Big Lots": 25_000,

    # ── Convenience / Gas ───────────────────────────────────────────────
    "7-Eleven": 1_800,
    "7-11": 1_800,
    "Circle K": 2_800,
    "Speedway": 3_000,
    "Casey's": 3_500,
    "Casey's General Store": 3_500,
    "Kum & Go": 4_500,
    "Buc-ee's": 50_000,
    "Love's": 12_000,
    "Love's Travel Stop": 12_000,
    "Pilot": 11_000,
    "Pilot Flying J": 11_000,
    "Flying J": 11_000,
    "RaceTrac": 4_500,
    "Raceway": 2_500,
    "Chevron": 2_500,
    "Shell": 2_500,
    "BP": 2_500,
    "Exxon": 2_500,
    "Mobil": 2_500,
    "Sunoco": 2_500,
    "Marathon": 2_500,
    "Citgo": 2_500,
    "Phillips 66": 2_500,
    "Valero": 2_500,
    "Murphy USA": 1_200,
    "Cumberland Farms": 3_500,
    "Royal Farms": 5_000,
    "Kwik Trip": 5_000,

    # ── Auto ────────────────────────────────────────────────────────────
    "AutoZone": 7_000,
    "O'Reilly": 7_000,
    "O'Reilly Auto Parts": 7_000,
    "OReilly": 7_000,
    "Advance Auto Parts": 7_200,
    "Advance Auto": 7_200,
    "NAPA Auto Parts": 7_000,
    "NAPA": 7_000,
    "Jiffy Lube": 2_500,
    "Valvoline": 2_200,
    "Firestone": 7_000,
    "Pep Boys": 12_000,
    "Discount Tire": 6_000,
    "Mavis": 5_000,
    "Meineke": 3_500,
    "Midas": 3_500,
    "Goodyear": 5_500,
    "Caliber Collision": 10_000,
    "Safelite": 5_000,
    "Take 5 Oil Change": 1_800,
    "CarMax": 55_000,

    # ── Department Store ────────────────────────────────────────────────
    "Macy's": 150_000,
    "Macys": 150_000,
    "Nordstrom": 140_000,
    "JCPenney": 100_000,
    "JC Penney": 100_000,
    "Kohl's": 88_000,
    "Kohls": 88_000,
    "Dillard's": 120_000,
    "Dillards": 120_000,
    "Neiman Marcus": 100_000,
    "Saks Fifth Avenue": 100_000,
    "Bloomingdale's": 120_000,
    "Belk": 80_000,

    # ── Pet ──────────────────────────────────────────────────────────────
    "PetSmart": 19_000,
    "Petco": 13_000,
    "Pet Supplies Plus": 8_000,

    # ── Fitness ──────────────────────────────────────────────────────────
    "Planet Fitness": 20_000,
    "LA Fitness": 45_000,
    "Anytime Fitness": 5_000,
    "Gold's Gym": 30_000,
    "Golds Gym": 30_000,
    "24 Hour Fitness": 35_000,
    "Orangetheory Fitness": 2_800,
    "Orangetheory": 2_800,
    "CrossFit": 3_000,
    "Equinox": 35_000,
    "YMCA": 50_000,
    "Lifetime Fitness": 110_000,
    "Life Time": 110_000,
    "Crunch Fitness": 22_000,
    "Crunch": 22_000,
    "Snap Fitness": 4_000,
    "F45": 1_800,
    "Pure Barre": 1_500,
    "Club Pilates": 2_000,
    "Cycle Bar": 2_200,

    # ── Bank ─────────────────────────────────────────────────────────────
    "Chase": 5_000,
    "JPMorgan Chase": 5_000,
    "Bank of America": 5_000,
    "Wells Fargo": 5_000,
    "Citibank": 5_000,
    "US Bank": 4_500,
    "PNC Bank": 4_500,
    "PNC": 4_500,
    "TD Bank": 4_500,
    "Capital One": 4_000,
    "Regions Bank": 4_500,
    "Truist": 4_500,
    "Fifth Third Bank": 4_500,
    "Huntington Bank": 4_500,
    "KeyBank": 4_000,
    "M&T Bank": 4_000,
    "BMO Harris": 4_500,

    # ── Sporting Goods / Outdoor ─────────────────────────────────────────
    "Dick's Sporting Goods": 50_000,
    "Dicks Sporting Goods": 50_000,
    "Academy Sports": 65_000,
    "Academy Sports + Outdoors": 65_000,
    "REI": 25_000,
    "Bass Pro Shops": 125_000,
    "Cabela's": 80_000,
    "Cabelas": 80_000,
    "Scheels": 200_000,
    "Golf Galaxy": 15_000,

    # ── Office / Craft / Book ────────────────────────────────────────────
    "Staples": 18_000,
    "Office Depot": 20_000,
    "OfficeMax": 20_000,
    "FedEx Office": 4_000,
    "UPS Store": 1_400,
    "The UPS Store": 1_400,
    "Michaels": 21_000,
    "Hobby Lobby": 55_000,
    "Joann Fabrics": 15_000,
    "JOANN": 15_000,
    "Barnes & Noble": 25_000,
    "Barnes and Noble": 25_000,
    "Half Price Books": 15_000,

    # ── Beauty / Personal Care ───────────────────────────────────────────
    "Ulta": 10_000,
    "Ulta Beauty": 10_000,
    "Sephora": 5_000,
    "Sally Beauty": 2_200,
    "Great Clips": 1_200,
    "Supercuts": 1_200,
    "Sport Clips": 1_200,
    "Bath & Body Works": 3_000,
    "GNC": 1_800,
    "Vitamin Shoppe": 4_000,
    "The Vitamin Shoppe": 4_000,

    # ── Wireless / Phone ─────────────────────────────────────────────────
    "Cricket Wireless": 1_400,
    "Metro by T-Mobile": 1_400,
    "MetroPCS": 1_400,
    "Boost Mobile": 1_200,

    # ── Rental / Services ────────────────────────────────────────────────
    "U-Haul": 5_000,
    "Public Storage": 10_000,
    "Extra Space Storage": 10_000,
    "CubeSmart": 8_000,
    "Life Storage": 9_000,

    # ── Jewelry / Watches ────────────────────────────────────────────────
    "Kay Jewelers": 2_000,
    "Zales": 2_000,
    "Jared": 5_000,
    "Tiffany & Co": 5_000,
    "Pandora": 1_200,

    # ── Eyewear / Vision ─────────────────────────────────────────────────
    "LensCrafters": 3_000,
    "Visionworks": 2_500,
    "Pearle Vision": 2_500,
    "Warby Parker": 1_800,

    # ── Misc Retail ──────────────────────────────────────────────────────
    "Party City": 12_000,
    "Spirit Halloween": 15_000,
    "Tuesday Morning": 10_000,
    "Ollie's Bargain Outlet": 30_000,
    "Ollies": 30_000,
    "Costco Business Center": 120_000,
}

# ---------------------------------------------------------------------------
# Category lookup  (OSM-derived categories -> average square footage)
# ---------------------------------------------------------------------------

CATEGORY_SQFT: dict[str, float] = {
    "grocery": 45_000,
    "convenience": 2_500,
    "department": 100_000,
    "apparel": 12_000,
    "electronics": 8_000,
    "furniture": 30_000,
    "hardware": 25_000,
    "food_specialty": 3_000,
    "restaurant": 5_000,
    "cafe": 1_800,
    "fast_food": 3_000,
    "bar": 3_000,
    "bank": 5_000,
    "pharmacy": 11_000,
    "health": 3_000,
    "personal_care": 1_500,
    "auto_services": 3_000,
    "auto_dealership": 20_000,
    "gas_station": 2_500,
    "supermarket": 55_000,
    "other": 3_000,
}

_DEFAULT_SQFT: float = 3_000.0

# ---------------------------------------------------------------------------
# Internal: build a case-folded index once at import time
# ---------------------------------------------------------------------------

_BRAND_INDEX: dict[str, float] = {k.casefold(): v for k, v in BRAND_SQFT.items()}


def _normalize(text: str) -> str:
    """Lower-case and strip common punctuation for fuzzy-ish matching."""
    return (
        text.casefold()
        .replace("'", "")
        .replace("\u2019", "")   # right single quote
        .replace(".", "")
        .replace(",", "")
        .replace("-", " ")
        .strip()
    )


# Pre-compute a normalized index for prefix / fuzzy matching.
_BRAND_NORM: dict[str, float] = {_normalize(k): v for k, v in BRAND_SQFT.items()}


def _match_brand(name_or_brand: str) -> Optional[float]:
    """Try to match a string to a known brand.

    Strategy (in order):
    1. Exact case-folded match on the full string.
    2. Exact match after punctuation normalization.
    3. Check if any known brand is a prefix of the input (longest wins).
    4. Check if the input is a prefix of any known brand (longest wins).
    """
    cf = name_or_brand.casefold()
    if cf in _BRAND_INDEX:
        return _BRAND_INDEX[cf]

    norm = _normalize(name_or_brand)
    if norm in _BRAND_NORM:
        return _BRAND_NORM[norm]

    # Prefix: brand name is a prefix of the input  ("McDonald's #1234")
    best: Optional[tuple[int, float]] = None
    for key, sqft in _BRAND_NORM.items():
        if norm.startswith(key):
            if best is None or len(key) > best[0]:
                best = (len(key), sqft)
        elif key.startswith(norm):
            if best is None or len(norm) > best[0]:
                best = (len(norm), sqft)
    if best is not None:
        return best[1]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_sqft(
    name: Optional[str] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
) -> tuple[float, str]:
    """Return ``(square_footage, source)`` for a single store.

    Parameters
    ----------
    name : str, optional
        The store name (e.g. ``"McDonald's #12345"``).
    brand : str, optional
        The brand tag (e.g. ``"McDonald's"``).
    category : str, optional
        An OSM-derived category key (e.g. ``"fast_food"``).

    Returns
    -------
    tuple[float, str]
        ``(sqft, source)`` where *source* is one of
        ``"brand"``, ``"category"``, or ``"default"``.
    """
    # 1. Try brand field first (most specific).
    if brand:
        result = _match_brand(brand)
        if result is not None:
            return (result, "brand")

    # 2. Try store name (often contains brand).
    if name:
        result = _match_brand(name)
        if result is not None:
            return (result, "brand")

    # 3. Fall back to category.
    if category:
        cat_lower = category.casefold().strip()
        if cat_lower in CATEGORY_SQFT:
            return (CATEGORY_SQFT[cat_lower], "category")

    # 4. Hard default.
    return (_DEFAULT_SQFT, "default")


def estimate_store_size(stores: list[Store]) -> list[Store]:
    """Estimate square footage for each store that lacks it.

    The function **mutates** each :class:`Store` in place:

    * ``store.square_footage`` is set to the estimated value (only when the
      current value is ``0.0``; existing non-zero values are preserved).
    * ``store.attributes["sqft_source"]`` is set to ``"brand"``,
      ``"category"``, or ``"default"`` to indicate how the value was derived.

    Parameters
    ----------
    stores : list[Store]
        Store objects (see :mod:`gravity.data.schema`).

    Returns
    -------
    list[Store]
        The same list, mutated in place, for convenient chaining.
    """
    for store in stores:
        if store.square_footage > 0:
            store.attributes["sqft_source"] = "measured"
            continue

        sqft, source = estimate_sqft(
            name=store.name,
            brand=store.brand,
            category=store.category,
        )
        store.square_footage = sqft
        store.attributes["sqft_source"] = source

    return stores
