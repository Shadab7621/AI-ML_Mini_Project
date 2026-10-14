// Readable names for the one-letter codes in the UCI Mushroom dataset
// (from the dataset's attribute documentation). The other datasets need none.

const COLORS = {
  n: "brown", b: "buff", c: "cinnamon", g: "gray", r: "green", p: "pink",
  u: "purple", e: "red", w: "white", y: "yellow", k: "black", h: "chocolate",
  o: "orange",
}
const SURFACE = { f: "fibrous", g: "grooves", y: "scaly", s: "smooth", k: "silky" }

const VALUE_NAMES: Record<string, Record<string, string>> = {
  type: { e: "edible", p: "poisonous" },
  cap_shape: { b: "bell", c: "conical", x: "convex", f: "flat", k: "knobbed", s: "sunken" },
  cap_surface: SURFACE,
  cap_color: COLORS,
  bruises: { t: "bruises", f: "no bruises" },
  odor: {
    a: "almond", l: "anise", c: "creosote", y: "fishy", f: "foul", m: "musty",
    n: "none", p: "pungent", s: "spicy",
  },
  gill_attachment: { a: "attached", d: "descending", f: "free", n: "notched" },
  gill_spacing: { c: "close", w: "crowded", d: "distant" },
  gill_size: { b: "broad", n: "narrow" },
  gill_color: COLORS,
  stalk_shape: { e: "enlarging", t: "tapering" },
  stalk_root: {
    b: "bulbous", c: "club", u: "cup", e: "equal", z: "rhizomorphs", r: "rooted",
    "?": "missing",
  },
  stalk_surface_above_ring: SURFACE,
  stalk_surface_below_ring: SURFACE,
  stalk_color_above_ring: COLORS,
  stalk_color_below_ring: COLORS,
  veil_type: { p: "partial", u: "universal" },
  veil_color: COLORS,
  ring_number: { n: "none", o: "one", t: "two" },
  ring_type: {
    c: "cobwebby", e: "evanescent", f: "flaring", l: "large", n: "none",
    p: "pendant", s: "sheathing", z: "zone",
  },
  spore_print_color: COLORS,
  population: {
    a: "abundant", c: "clustered", n: "numerous", s: "scattered", v: "several",
    y: "solitary",
  },
  habitat: {
    g: "grasses", l: "leaves", m: "meadows", p: "paths", u: "urban", w: "waste",
    d: "woods",
  },
}

/** "stalk_surface_above_ring" -> "stalk surface above ring" */
export const featureLabel = (feature: string) => feature.replaceAll("_", " ")

/**
 * ("mushrooms", "odor", "f") -> "foul". Only the mushroom data uses one-letter codes;
 * values of other datasets, and unknown codes, are returned as they are.
 */
export const valueLabel = (dataset: string, feature: string, code: string) =>
  dataset === "mushrooms" ? (VALUE_NAMES[feature]?.[code] ?? code) : code
