#!/usr/bin/env -S gawk -f
# The evolved UK-postcode rule (typed GEP regex kingdom, run newA28 seed 7015, phylu
# docs/reports/2026-10-20_regex_typed_gep.md):  (?:.)*[ -'](?:.)*[=-i](?:[%-(])?
# as a POSIX ERE matched against the whole field (the fit used full-string match).
# Run under LC_ALL=C so the ranges are byte ranges, as in the evolved rule.
#
# Usage: LC_ALL=C gawk -f uk_postcode_rule.awk file        (one value per line: prints matches)
#        LC_ALL=C gawk -v col=1 -F'\t' -f uk_postcode_rule.awk file.tsv   (test one column)
BEGIN { if (col == "") col = 0 }
{
    v = (col == 0) ? $0 : $col
    if (v ~ /^.*[ -'].*[=-i][%-(]?$/) print
}
