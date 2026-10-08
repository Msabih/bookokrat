//! Right-to-left text support for the table of contents.
//!
//! Two independent pieces:
//!
//! * **Direction detection** ([`DirectionCounter`], [`detect_direction`]): decides
//!   whether a document is primarily right-to-left by counting strong RTL
//!   characters (Arabic script incl. Urdu/Persian letters and presentation forms,
//!   Hebrew, Syriac, Thaana, NKo, ...) against strong LTR characters (Latin,
//!   Greek, Cyrillic, CJK, ...), using the Unicode bidi classes.
//! * **Visual rendering** ([`VisualText::shaped`]): most terminals (wezterm with
//!   its default `bidi_enabled = false`, kitty, ...) draw cells strictly left to
//!   right in the order they are written and do no Arabic shaping. To make
//!   Urdu/Arabic titles readable there, the text is contextually shaped into
//!   Arabic Presentation Forms (in logical order) and then reordered into visual
//!   order with the Unicode Bidirectional Algorithm. Combining marks stay
//!   attached to their base letter so every visual cluster occupies exactly the
//!   cells its base character needs.
//!
//! Everything here is pure and allocation-light; callers are expected to cache
//! the result (the TOC shapes each title once per TOC load, not per frame).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::OnceLock;

use ar_reshaper::form::Forms;
use unicode_bidi::{BidiClass, Level, ParagraphBidiInfo, bidi_class};
use unicode_width::UnicodeWidthChar;

/// Main writing direction of a document or a piece of text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextDirection {
    #[default]
    Ltr,
    Rtl,
}

impl TextDirection {
    pub fn is_rtl(self) -> bool {
        self == TextDirection::Rtl
    }
}

/// Strong direction of a single character, or `None` for neutral/weak
/// characters (digits, punctuation, spaces, combining marks, ...).
pub fn strong_direction(c: char) -> Option<TextDirection> {
    match bidi_class(c) {
        BidiClass::R | BidiClass::AL => Some(TextDirection::Rtl),
        BidiClass::L => Some(TextDirection::Ltr),
        _ => None,
    }
}

/// Whether the text contains at least one strong right-to-left character.
pub fn contains_rtl(text: &str) -> bool {
    text.chars()
        .any(|c| strong_direction(c) == Some(TextDirection::Rtl))
}

/// Incremental counter of strong RTL vs. strong LTR characters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DirectionCounter {
    pub rtl: usize,
    pub ltr: usize,
}

impl DirectionCounter {
    pub fn add_char(&mut self, c: char) {
        match strong_direction(c) {
            Some(TextDirection::Rtl) => self.rtl += 1,
            Some(TextDirection::Ltr) => self.ltr += 1,
            None => {}
        }
    }

    pub fn add_str(&mut self, text: &str) {
        text.chars().for_each(|c| self.add_char(c));
    }

    /// Number of strong characters seen so far.
    pub fn strong_total(&self) -> usize {
        self.rtl + self.ltr
    }

    /// The majority direction; `None` when no strong character was seen.
    /// RTL has to strictly outnumber LTR (a tie stays LTR).
    pub fn direction(&self) -> Option<TextDirection> {
        if self.strong_total() == 0 {
            None
        } else if self.rtl > self.ltr {
            Some(TextDirection::Rtl)
        } else {
            Some(TextDirection::Ltr)
        }
    }
}

/// Detect the main direction of a set of texts (e.g. all TOC titles).
/// Returns `None` when the texts contain no strong directional character.
pub fn detect_direction<'a, I>(texts: I) -> Option<TextDirection>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut counter = DirectionCounter::default();
    for text in texts {
        counter.add_str(text);
    }
    counter.direction()
}

/// Direction implied by a BCP 47 language tag (e.g. EPUB `dc:language`).
/// Returns `None` for an empty tag.
pub fn direction_for_language(tag: &str) -> Option<TextDirection> {
    const RTL_LANGUAGES: &[&str] = &[
        "ar", "arc", "ckb", "dv", "fa", "glk", "he", "iw", "khw", "ks", "ku", "mzn", "nqo", "pnb",
        "ps", "sd", "syc", "syr", "ug", "ur", "yi",
    ];
    const RTL_SCRIPTS: &[&str] = &["arab", "hebr", "syrc", "thaa", "nkoo", "adlm", "rohg"];

    let tag = tag.trim();
    if tag.is_empty() {
        return None;
    }
    let mut subtags = tag.split(['-', '_']).map(|s| s.to_ascii_lowercase());
    let primary = subtags.next().unwrap_or_default();
    let script = subtags.find(|s| s.len() == 4);
    let rtl = match script {
        Some(script) => RTL_SCRIPTS.contains(&script.as_str()),
        None => RTL_LANGUAGES.contains(&primary.as_str()),
    };
    Some(if rtl {
        TextDirection::Rtl
    } else {
        TextDirection::Ltr
    })
}

/// One display cluster: a base character plus its combining marks, which a
/// terminal draws in the cell(s) of the base character.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisualCluster {
    /// The characters to output (base first, then marks).
    pub text: String,
    /// Range of source (logical) character indices this cluster represents.
    pub source: Range<usize>,
}

/// Text prepared for display: clusters in the order they must be written to
/// the terminal, each mapped back to the logical characters it came from (so
/// search matches on the logical text can be highlighted).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VisualText {
    pub clusters: Vec<VisualCluster>,
}

impl VisualText {
    /// The text in logical order, only grouped into clusters (for terminals
    /// that apply bidi and shaping themselves).
    pub fn logical(text: &str) -> Self {
        let chars: Vec<ShapedChar> = text
            .chars()
            .map(sanitize_char)
            .enumerate()
            .filter(|(_, c)| !is_invisible_control(*c))
            .map(|(i, c)| ShapedChar {
                ch: c,
                source: i..i + 1,
            })
            .collect();
        let clusters = clusterize(&chars)
            .into_iter()
            .map(|range| make_cluster(&chars[range], false))
            .collect();
        Self { clusters }
    }

    /// Shape Arabic-script letters into presentation forms and reorder the
    /// text into visual (left-to-right cell) order for a paragraph of the
    /// given base direction.
    pub fn shaped(text: &str, base: TextDirection) -> Self {
        let shaped = shape_arabic(&prepare_for_shaping(text));
        if shaped.is_empty() {
            return Self::default();
        }

        let shaped_text: String = shaped.iter().map(|s| s.ch).collect();
        let base_level = match base {
            TextDirection::Ltr => Level::ltr(),
            TextDirection::Rtl => Level::rtl(),
        };
        let bidi = ParagraphBidiInfo::new(&shaped_text, Some(base_level));
        let levels = bidi.reordered_levels_per_char(0..shaped_text.len());

        // Drop zero-width controls (ZWJ/ZWNJ, bidi marks) now that they have
        // served joining and bidi resolution: terminals may draw them as boxes.
        let (chars, levels): (Vec<ShapedChar>, Vec<Level>) = shaped
            .into_iter()
            .zip(levels)
            .filter(|(s, _)| !is_invisible_control(s.ch))
            .unzip();

        let cluster_ranges = clusterize(&chars);
        let cluster_levels: Vec<Level> = cluster_ranges
            .iter()
            .map(|range| levels[range.start])
            .collect();
        let order = unicode_bidi::BidiInfo::reorder_visual(&cluster_levels);
        let clusters = order
            .into_iter()
            .map(|idx| {
                let range = cluster_ranges[idx].clone();
                make_cluster(&chars[range], cluster_levels[idx].is_rtl())
            })
            .collect();
        Self { clusters }
    }

    /// The display string.
    pub fn text(&self) -> String {
        self.clusters.iter().map(|c| c.text.as_str()).collect()
    }

    /// Terminal display width in cells.
    pub fn width(&self) -> usize {
        self.clusters
            .iter()
            .map(|c| unicode_width::UnicodeWidthStr::width(c.text.as_str()))
            .sum()
    }
}

/// A shaped character with the logical source characters it represents.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ShapedChar {
    ch: char,
    source: Range<usize>,
}

/// Line breaks and tabs are never wanted inside a one-line title.
fn sanitize_char(c: char) -> char {
    if c.is_control() { ' ' } else { c }
}

/// Sanitized characters paired with their logical index in the source text.
/// Letters that have no presentation forms of their own but decompose into
/// one that has (heh goal with hamza above) are decomposed so they still join.
fn prepare_for_shaping(text: &str) -> Vec<(char, usize)> {
    let mut out = Vec::with_capacity(text.len());
    for (i, c) in text.chars().map(sanitize_char).enumerate() {
        match c {
            '\u{06C2}' => {
                out.push(('\u{06C1}', i));
                out.push(('\u{0654}', i));
            }
            _ => out.push((c, i)),
        }
    }
    out
}

const ZWNJ: char = '\u{200C}';
const ZWJ: char = '\u{200D}';
const LAM: char = '\u{0644}';

/// Zero-width format characters that are dropped from the output.
fn is_invisible_control(c: char) -> bool {
    matches!(
        c,
        '\u{200B}'..='\u{200F}' | '\u{061C}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{FEFF}'
    )
}

/// Combining marks attach to the preceding base character.
fn is_combining(c: char) -> bool {
    !is_invisible_control(c) && !c.is_control() && c.width() == Some(0)
}

fn clusterize(chars: &[ShapedChar]) -> Vec<Range<usize>> {
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for (i, s) in chars.iter().enumerate() {
        match ranges.last_mut() {
            Some(last) if is_combining(s.ch) => last.end = i + 1,
            _ => ranges.push(i..i + 1),
        }
    }
    ranges
}

fn make_cluster(chars: &[ShapedChar], mirror: bool) -> VisualCluster {
    let mut text = String::with_capacity(chars.len() * 3);
    for (i, s) in chars.iter().enumerate() {
        let c = if i == 0 && mirror { mirrored(s.ch) } else { s.ch };
        text.push(c);
    }
    let start = chars.iter().map(|s| s.source.start).min().unwrap_or(0);
    let end = chars.iter().map(|s| s.source.end).max().unwrap_or(start);
    VisualCluster {
        text,
        source: start..end,
    }
}

/// Bidi mirroring (rule L4) for the paired punctuation that occurs in titles.
fn mirrored(c: char) -> char {
    match c {
        '(' => ')',
        ')' => '(',
        '[' => ']',
        ']' => '[',
        '{' => '}',
        '}' => '{',
        '<' => '>',
        '>' => '<',
        '«' => '»',
        '»' => '«',
        '‹' => '›',
        '›' => '‹',
        _ => c,
    }
}

/// Unicode Arabic joining types relevant for shaping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Joining {
    /// Joins on both sides (beh, seen, heh goal, ...).
    Dual,
    /// Joins only to the preceding letter (alef, dal, reh, waw, yeh barree, ...).
    Right,
    /// Forces joining on both sides without changing shape (tatweel, ZWJ).
    Causing,
    /// Skipped when determining joining (harakat and other marks).
    Transparent,
    /// Breaks joining (spaces, punctuation, Latin, hamza, ZWNJ, ...).
    None,
}

/// Presentation-form table from `ar-reshaper` (python-arabic-reshaper data),
/// which covers Arabic, Persian and Urdu letters.
fn letter_forms() -> &'static HashMap<char, Forms> {
    static FORMS: OnceLock<HashMap<char, Forms>> = OnceLock::new();
    FORMS.get_or_init(|| {
        ar_reshaper::letters::letters_db::LETTERS_ARABIC
            .iter()
            .copied()
            .collect()
    })
}

fn joining_type(c: char) -> Joining {
    if c == ZWJ || c == '\u{0640}' {
        return Joining::Causing;
    }
    if c == ZWNJ {
        return Joining::None;
    }
    if is_combining(c) || is_invisible_control(c) {
        return Joining::Transparent;
    }
    if let Some(forms) = letter_forms().get(&c) {
        return if forms.medial != '\0' || forms.initial != '\0' {
            Joining::Dual
        } else if forms.end != '\0' {
            Joining::Right
        } else {
            Joining::None
        };
    }
    // Letters without presentation forms that still take part in joining.
    match c {
        '\u{06CD}' | '\u{06CE}' | '\u{0750}'..='\u{0768}' | '\u{076D}'..='\u{0770}' => {
            Joining::Dual
        }
        '\u{06C3}' | '\u{06C4}' | '\u{06CA}' | '\u{06CF}' | '\u{06D5}' | '\u{06EE}' | '\u{06EF}' => {
            Joining::Right
        }
        _ => Joining::None,
    }
}

fn joins_to_following(j: Joining) -> bool {
    matches!(j, Joining::Dual | Joining::Causing)
}

fn joins_to_preceding(j: Joining) -> bool {
    matches!(j, Joining::Dual | Joining::Right | Joining::Causing)
}

/// Lam-alef ligatures: (alef variant, isolated, final).
fn lam_alef_ligature(alef: char) -> Option<(char, char)> {
    match alef {
        '\u{0622}' => Some(('\u{FEF5}', '\u{FEF6}')),
        '\u{0623}' => Some(('\u{FEF7}', '\u{FEF8}')),
        '\u{0625}' => Some(('\u{FEF9}', '\u{FEFA}')),
        '\u{0627}' => Some(('\u{FEFB}', '\u{FEFC}')),
        _ => None,
    }
}

/// Contextual shaping in logical order. Each output character records the
/// logical character indices it stands for (two for a lam-alef ligature).
fn shape_arabic(input: &[(char, usize)]) -> Vec<ShapedChar> {
    let chars: Vec<char> = input.iter().map(|(c, _)| *c).collect();
    let source = |i: usize| input[i].1..input[i].1 + 1;
    let types: Vec<Joining> = chars.iter().map(|&c| joining_type(c)).collect();

    let prev_joining = |i: usize| -> Option<Joining> {
        types[..i]
            .iter()
            .rev()
            .find(|j| **j != Joining::Transparent)
            .copied()
    };
    let next_index = |i: usize| -> Option<usize> {
        (i + 1..chars.len()).find(|&k| types[k] != Joining::Transparent)
    };

    let mut out = Vec::with_capacity(chars.len());
    let mut skip_alef: Option<usize> = None;
    for (i, &c) in chars.iter().enumerate() {
        if skip_alef == Some(i) {
            continue;
        }
        let jt = types[i];
        if jt == Joining::Transparent {
            out.push(ShapedChar {
                ch: c,
                source: source(i),
            });
            continue;
        }

        let joins_prev = joins_to_preceding(jt) && prev_joining(i).is_some_and(joins_to_following);
        let next = next_index(i);

        if c == LAM {
            if let Some(n) = next {
                if let Some((isolated, final_form)) = lam_alef_ligature(chars[n]) {
                    // Marks between lam and alef stay with the ligature.
                    let ch = if joins_prev { final_form } else { isolated };
                    out.push(ShapedChar {
                        ch,
                        source: input[i].1..input[n].1 + 1,
                    });
                    for (k, &mark) in chars.iter().enumerate().take(n).skip(i + 1) {
                        out.push(ShapedChar {
                            ch: mark,
                            source: source(k),
                        });
                    }
                    skip_alef = Some(n);
                    continue;
                }
            }
        }

        let joins_next =
            joins_to_following(jt) && next.is_some_and(|n| joins_to_preceding(types[n]));
        let ch = match letter_forms().get(&c) {
            Some(forms) if jt != Joining::Causing => {
                let form = match (joins_prev, joins_next) {
                    (true, true) => forms.medial,
                    (true, false) => forms.end,
                    (false, true) => forms.initial,
                    (false, false) => forms.isolated,
                };
                if form == '\0' { forms.isolated } else { form }
            }
            _ => c,
        };
        out.push(ShapedChar {
            ch,
            source: source(i),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visual(text: &str, base: TextDirection) -> String {
        VisualText::shaped(text, base).text()
    }

    #[test]
    fn detects_urdu_titles_as_rtl() {
        let titles = ["مقدمات", "مسئلۂ تنزیہ کا تعارف", "Chapter 1: ابن تیمیہ"];
        assert_eq!(detect_direction(titles), Some(TextDirection::Rtl));
    }

    #[test]
    fn detects_english_titles_as_ltr() {
        let titles = ["Introduction", "1. Getting started", "Appendix: العربية"];
        assert_eq!(detect_direction(titles), Some(TextDirection::Ltr));
    }

    #[test]
    fn detects_hebrew_syriac_thaana_nko_as_rtl() {
        for text in ["שלום עולם", "ܫܠܡܐ", "ދިވެހި", "ߒߞߏ"] {
            assert_eq!(detect_direction([text]), Some(TextDirection::Rtl), "{text}");
        }
    }

    #[test]
    fn presentation_forms_count_as_rtl() {
        assert_eq!(
            detect_direction(["\u{FEB3}\u{FEFC}\u{FEE1}"]),
            Some(TextDirection::Rtl)
        );
        assert_eq!(detect_direction(["\u{FB8A}"]), Some(TextDirection::Rtl));
    }

    #[test]
    fn neutral_only_text_has_no_direction() {
        assert_eq!(detect_direction(["123 - 456", "۷۰۵", "", "()"]), None);
        assert_eq!(detect_direction(std::iter::empty::<&str>()), None);
    }

    #[test]
    fn tie_stays_ltr() {
        let mut counter = DirectionCounter::default();
        counter.add_str("ab");
        counter.add_str("اب");
        assert_eq!(counter.rtl, 2);
        assert_eq!(counter.ltr, 2);
        assert_eq!(counter.direction(), Some(TextDirection::Ltr));
    }

    #[test]
    fn language_tags_map_to_direction() {
        assert_eq!(direction_for_language("ur"), Some(TextDirection::Rtl));
        assert_eq!(direction_for_language("ar-SA"), Some(TextDirection::Rtl));
        assert_eq!(direction_for_language("he"), Some(TextDirection::Rtl));
        assert_eq!(direction_for_language("en-US"), Some(TextDirection::Ltr));
        assert_eq!(direction_for_language("az-Arab"), Some(TextDirection::Rtl));
        assert_eq!(direction_for_language("ku-Latn"), Some(TextDirection::Ltr));
        assert_eq!(direction_for_language("  "), None);
    }

    #[test]
    fn shapes_arabic_word_contextually() {
        // سلام: seen initial, lam-alef final ligature, meem isolated
        assert_eq!(
            visual("سلام", TextDirection::Rtl),
            "\u{FEE1}\u{FEFC}\u{FEB3}"
        );
    }

    #[test]
    fn shapes_urdu_letters() {
        // پاکستان: peh initial, alef final, keheh initial, seen medial,
        // teh medial, alef final, noon isolated — written in visual order.
        let expected: String = [
            '\u{FEE5}', '\u{FE8E}', '\u{FE98}', '\u{FEB4}', '\u{FB90}', '\u{FE8E}', '\u{FB58}',
        ]
        .iter()
        .collect();
        assert_eq!(visual("پاکستان", TextDirection::Rtl), expected);
    }

    #[test]
    fn shapes_urdu_specific_letters() {
        // ٹ ڈ ڑ ں ہ ھ ے ی گ چ ژ in joining contexts
        let cases = [
            ("ٹب", '\u{FB68}'),  // tteh initial
            ("بڈ", '\u{FB89}'),  // ddal final
            ("بڑ", '\u{FB8D}'),  // rreh final
            ("بں", '\u{FB9F}'),  // noon ghunna final
            ("ہب", '\u{FBA8}'),  // heh goal initial
            ("بھب", '\u{FBAD}'), // heh doachashmee medial
            ("بے", '\u{FBAF}'),  // yeh barree final
            ("بیب", '\u{FBFF}'), // farsi yeh medial
            ("گب", '\u{FB94}'),  // gaf initial
            ("چب", '\u{FB7C}'),  // tcheh initial
            ("بژ", '\u{FB8B}'),  // jeh final
        ];
        for (text, form) in cases {
            let shaped = visual(text, TextDirection::Rtl);
            assert!(shaped.contains(form), "{text}: {shaped:?} lacks {form:?}");
        }
    }

    #[test]
    fn diacritics_stay_attached_to_their_base() {
        // بُت with damma on beh: the mark must follow the beh in output order
        let vt = VisualText::shaped("بُت", TextDirection::Rtl);
        assert_eq!(vt.clusters.len(), 2);
        assert_eq!(vt.clusters[1].text, "\u{FE91}\u{064F}");
        assert_eq!(vt.clusters[1].source, 0..2);
        assert_eq!(vt.width(), 2);
    }

    #[test]
    fn marks_are_transparent_for_joining() {
        // tashdid between two behs must not break the join
        let shaped = visual("بّب", TextDirection::Rtl);
        assert!(shaped.contains('\u{FE91}'), "{shaped:?}");
        assert!(shaped.contains('\u{FE90}'), "{shaped:?}");
    }

    #[test]
    fn zwnj_breaks_joining_and_is_dropped() {
        let shaped = visual("ب\u{200C}ب", TextDirection::Rtl);
        assert_eq!(shaped, "\u{FE8F}\u{FE8F}");
    }

    #[test]
    fn numbers_and_latin_keep_ltr_order_in_rtl_paragraph() {
        // "باب 12" -> digits stay "12" and end up on the visual left
        let shaped = visual("باب 12", TextDirection::Rtl);
        assert!(shaped.starts_with("12 "), "{shaped:?}");
        let shaped = visual("ابن تیمیہ (Ibn Taymiyya)", TextDirection::Rtl);
        assert!(shaped.starts_with("(Ibn Taymiyya) "), "{shaped:?}");
    }

    #[test]
    fn rtl_word_in_ltr_paragraph_is_reversed_in_place() {
        let shaped = visual("Intro سلام end", TextDirection::Ltr);
        assert_eq!(shaped, "Intro \u{FEE1}\u{FEFC}\u{FEB3} end");
    }

    #[test]
    fn latin_text_is_unchanged() {
        assert_eq!(visual("Plain title", TextDirection::Ltr), "Plain title");
        assert_eq!(
            VisualText::logical("Plain title").text(),
            "Plain title"
        );
    }

    #[test]
    fn width_matches_cell_count() {
        let vt = VisualText::shaped("فتاویٰ دمشقیہ: ۷۰۵ھ سے پہلے", TextDirection::Rtl);
        let text = vt.text();
        assert_eq!(vt.width(), unicode_width::UnicodeWidthStr::width(text.as_str()));
        // every cluster is exactly one cell wide
        assert!(
            vt.clusters
                .iter()
                .all(|c| unicode_width::UnicodeWidthStr::width(c.text.as_str()) == 1)
        );
    }

    #[test]
    fn source_ranges_cover_logical_text() {
        let title = "سلام دنیا";
        let vt = VisualText::shaped(title, TextDirection::Rtl);
        let mut covered = vec![false; title.chars().count()];
        for cluster in &vt.clusters {
            for i in cluster.source.clone() {
                covered[i] = true;
            }
        }
        assert!(covered.iter().all(|c| *c));
        // lam-alef ligature covers both lam and alef
        assert!(vt.clusters.iter().any(|c| c.source == (1..3)));
    }

    #[test]
    fn brackets_are_mirrored_in_rtl_runs() {
        let shaped = visual("(باب)", TextDirection::Rtl);
        assert!(shaped.starts_with('('), "{shaped:?}");
        assert!(shaped.ends_with(')'), "{shaped:?}");
    }

    #[test]
    fn heh_goal_with_hamza_joins_via_decomposition() {
        // مسئلۂ: lam joins the following heh goal, which takes its final form
        let vt = VisualText::shaped("لۂ", TextDirection::Rtl);
        assert_eq!(vt.text(), "\u{FBA7}\u{0654}\u{FEDF}");
        assert_eq!(vt.clusters[0].source, 1..2);
    }

    #[test]
    fn control_characters_are_sanitized() {
        let shaped = visual("ب\nب", TextDirection::Rtl);
        assert_eq!(shaped, "\u{FE8F} \u{FE8F}");
        assert_eq!(visual("\u{200F}ب", TextDirection::Rtl), "\u{FE8F}");
    }
}
