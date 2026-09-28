"""Maps LIVE's ~256 fine-grained concept tags to 15 broad topic categories.

LIVE by Po-Shen Loh tags every problem with one or more fine-grained
concepts (e.g. "Vieta's Formulas", "stars and bars"), grouped on their
site into 8 subjects (arithmetic, algebra, number theory, geometry,
combinatorics, probability and statistics, logic, problem-solving
techniques). That's too coarse for algebra/geometry (48 and 86 concepts
respectively) and too fine for building a per-topic ability model with
usable amounts of data per topic, so we re-bucket into 15 categories
sized so each has a reasonable number of historical AMC 10 problems.

The raw scrape this mapping was derived from is kept at
data/raw/live_concepts_taxonomy_raw.json for auditability.
"""
from __future__ import annotations

BROAD_CATEGORIES: list[str] = [
    "Arithmetic & Number Sense",
    "Algebra: Equations & Polynomials",
    "Algebra: Sequences, Series & Functions",
    "Algebra: Inequalities & Advanced Techniques",
    "Number Theory: Divisibility, Modular & Bases",
    "Number Theory: Primes & Number Properties",
    "Combinatorics: Basic Counting",
    "Combinatorics: Advanced Structures",
    "Probability",
    "Statistics & Data",
    "Geometry: Triangles & Polygons",
    "Geometry: Circles & Curves",
    "Geometry: Coordinate, Transformations & Vectors",
    "Geometry: Solid (3D) Geometry",
    "Logic & Problem-Solving Strategies",
]

# --- Arithmetic & Number Sense -------------------------------------------
_ARITHMETIC_NUMBER_SENSE = [
    "clock", "date and time", "decimal", "distance rate and time",
    "distributive property", "estimation", "exponent", "factorial",
    "fraction", "money", "order of operations", "percentage", "place value",
    "rate", "ratio and proportion", "relative speed", "repeating decimal",
    "unit conversion", "whole number operations",
]

# --- Algebra: Equations & Polynomials -------------------------------------
_ALGEBRA_EQUATIONS = [
    "algebraic manipulation", "completing the square", "custom operation",
    "difference of squares", "factoring", "linear equation", "mixture",
    "polynomial", "quadratic", "radical", "rational equation",
    "rationalizing denominator", "Simon's Favorite Factoring Trick",
    "substitution", "sum and difference of cubes", "system of equations",
    "zero product property", "ages", "determinant", "matrix",
    "partial fractions",
]

# --- Algebra: Sequences, Series & Functions -------------------------------
_ALGEBRA_SEQUENCES_FUNCTIONS = [
    "arithmetic sequence", "arithmetico-geometric series", "Fibonacci",
    "floor and ceiling functions", "function", "functional equation",
    "geometric sequence", "logarithm", "recursion", "slope", "summation",
    "sum of first n cubes", "sum of first n odd numbers",
    "sum of first n squares", "telescoping", "complex number",
    "roots of unity", "De Moivre's Theorem",
]

# --- Algebra: Inequalities & Advanced Techniques --------------------------
_ALGEBRA_ADVANCED = [
    "absolute value", "AM-GM Inequality", "calculus",
    "Cauchy-Schwarz Inequality", "inequality", "Newton's Sums",
    "symmetry (algebra)", "Vieta's Formulas", "binomial theorem",
    "number base",
]

# --- Number Theory: Divisibility, Modular & Bases -------------------------
_NT_DIVISIBILITY_MODULAR = [
    "Chinese Remainder Theorem", "divisibility", "Euler's Totient Function",
    "Fermat's Little Theorem", "greatest common divisor",
    "least common multiple", "modular arithmetic", "modular exponentiation",
    "multiple", "multiplicative order", "parity", "quadratic residue",
    "units digit", "trailing zeros",
]

# --- Number Theory: Primes & Number Properties ----------------------------
_NT_PRIMES_PROPERTIES = [
    "Chicken McNugget Theorem", "continued fraction", "digits",
    "Diophantine Equation", "factor", "factor counting",
    "Legendre's Formula", "palindrome", "perfect power", "perfect square",
    "power of 2", "prime", "prime factorization", "sum of factors",
    "triangular number",
]

# --- Combinatorics: Basic Counting ----------------------------------------
_COMBO_BASIC = [
    "basic counting", "circular arrangements", "circular counting", "combinations",
    "complementary counting", "counting integers in a range",
    "counting pairs", "fencepost counting", "multiplication principle",
    "multiset permutations", "permutations", "subsets",
    "counting shapes in figures", "counting regions",
    "counting intersections",
]

# --- Combinatorics: Advanced Structures -----------------------------------
_COMBO_ADVANCED = [
    "arrangements with restrictions", "bijection", "Burnside's Lemma",
    "Catalan Number", "combinatorial game", "derangements",
    "double counting", "generating functions", "graph theory",
    "inclusion-exclusion", "lattice paths", "partitions and compositions",
    "Pascal's Triangle", "parking functions", "pigeonhole principle", "recursive counting",
    "stars and bars", "tiling", "tree diagram", "Vandermonde's Convolution",
    "Venn Diagram",
]

# --- Probability -----------------------------------------------------------
_PROBABILITY = [
    "basic probability", "Bayes' Theorem", "binomial probability",
    "complementary probability", "conditional probability",
    "dice (probability)", "expected value", "geometric distribution",
    "geometric probability", "independent events", "random walk",
    "recursive probability", "sampling with replacement",
    "sampling without replacement",
]

# --- Statistics & Data ------------------------------------------------------
_STATISTICS = [
    "data and graph interpretation", "harmonic mean", "mean",
    "median (data)", "mode", "range", "weighted mean",
]

# --- Geometry: Triangles & Polygons -----------------------------------------
_GEO_TRIANGLES_POLYGONS = [
    "altitude", "angle bisector", "angle bisector theorem", "angle chasing",
    "angle sum", "area", "area decomposition", "area ratio",
    "Brahmagupta's Formula", "British Flag Theorem", "centroid", "Ceva's Theorem",
    "congruence (geometry)", "cyclic quadrilateral",
    "equiangular polygon", "equilateral triangle", "Euler's Polyhedron Formula",
    "Heron's Formula", "incircle, incenter, and inradius",
    "isosceles triangle", "kite", "law of cosines", "law of sines",
    "mass points", "median (geometry)", "midpoint", "parallelogram",
    "perimeter", "Pick's Theorem", "Ptolemy's Theorem", "Pythagorean Theorem",
    "Pythagorean Triple", "rectangle", "regular polygon", "rhombus",
    "right triangle", "similarity", "special right triangle",
    "Stewart's Theorem", "trapezoid", "triangle area", "triangle inequality",
    "square (geometry)", "diagonal", "parallel lines",
    "perpendicular bisector",
]

# --- Geometry: Circles & Curves ---------------------------------------------
_GEO_CIRCLES_CURVES = [
    "annulus", "arc", "chord", "circle", "circle area",
    "circumcircle, circumcenter, and circumradius", "circumference",
    "ellipse", "hyperbola", "inscribed angle", "parabola",
    "power of a point", "radical axis", "sector", "tangent circles",
    "tangent line", "trigonometric identity", "trigonometry",
]

# --- Geometry: Coordinate, Transformations & Vectors ------------------------
_GEO_COORDINATE_TRANSFORM = [
    "coordinate geometry", "distance formula", "homothety", "lattice point",
    "paper folding", "reflection (geometry)", "shoelace formula",
    "transformation", "vector", "Viviani's Theorem",
]

# --- Geometry: Solid (3D) Geometry ------------------------------------------
_GEO_SOLID = [
    "3D geometry", "cone", "cube geometry", "cylinder", "net (3D geometry)",
    "polyhedron", "power scaling of length, area, and volume", "pyramid",
    "rectangular prism", "sphere", "surface area", "volume",
]

# --- Logic & Problem-Solving Strategies -------------------------------------
_LOGIC_STRATEGIES = [
    "counterexample", "cryptarithm", "logical deduction", "magic square",
    "truth-tellers and liars", "bounding to limit cases", "casework",
    "extremal argument", "induction", "invariant", "optimization",
    "pairing and grouping", "pattern recognition", "process simulation",
    "small cases", "symmetry", "systematic listing", "work backwards",
]

_BUCKETS: dict[str, list[str]] = {
    "Arithmetic & Number Sense": _ARITHMETIC_NUMBER_SENSE,
    "Algebra: Equations & Polynomials": _ALGEBRA_EQUATIONS,
    "Algebra: Sequences, Series & Functions": _ALGEBRA_SEQUENCES_FUNCTIONS,
    "Algebra: Inequalities & Advanced Techniques": _ALGEBRA_ADVANCED,
    "Number Theory: Divisibility, Modular & Bases": _NT_DIVISIBILITY_MODULAR,
    "Number Theory: Primes & Number Properties": _NT_PRIMES_PROPERTIES,
    "Combinatorics: Basic Counting": _COMBO_BASIC,
    "Combinatorics: Advanced Structures": _COMBO_ADVANCED,
    "Probability": _PROBABILITY,
    "Statistics & Data": _STATISTICS,
    "Geometry: Triangles & Polygons": _GEO_TRIANGLES_POLYGONS,
    "Geometry: Circles & Curves": _GEO_CIRCLES_CURVES,
    "Geometry: Coordinate, Transformations & Vectors": _GEO_COORDINATE_TRANSFORM,
    "Geometry: Solid (3D) Geometry": _GEO_SOLID,
    "Logic & Problem-Solving Strategies": _LOGIC_STRATEGIES,
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace("’", "'")


_CONCEPT_TO_CATEGORY: dict[str, str] = {}
for _category, _concepts in _BUCKETS.items():
    for _c in _concepts:
        _CONCEPT_TO_CATEGORY[_normalize(_c)] = _category

UNCATEGORIZED = "Uncategorized"


def concept_to_category(concept: str) -> str:
    """Map a single fine-grained concept string to one of the 15 categories.

    Falls back to UNCATEGORIZED (rather than raising) for any concept LIVE
    might introduce later that isn't in our mapping yet, so ingestion never
    hard-fails on a new tag -- but the caller should log/count these so
    the mapping can be extended.
    """
    return _CONCEPT_TO_CATEGORY.get(_normalize(concept), UNCATEGORIZED)


def concepts_to_categories(concepts: list[str]) -> list[str]:
    """De-duplicated, order-preserving list of broad categories for a problem
    tagged with multiple fine-grained concepts (a problem may legitimately
    span more than one category, e.g. "modular arithmetic" + "casework")."""
    seen: list[str] = []
    for c in concepts:
        cat = concept_to_category(c)
        if cat not in seen:
            seen.append(cat)
    return seen


def coverage_report(all_concepts: set[str]) -> dict[str, list[str]]:
    """Given every distinct concept string seen during ingestion, return
    {"uncategorized": [...]} so validate_data.py can flag mapping gaps."""
    uncategorized = sorted(
        c for c in all_concepts if _normalize(c) not in _CONCEPT_TO_CATEGORY
    )
    return {"uncategorized": uncategorized}
