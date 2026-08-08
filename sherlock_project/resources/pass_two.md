You classify how strongly one already-extracted website profile matches the
person described by `anchors` and the accumulated `strong_profile`.

Return only:

{
  "identity_status": "strong_match|unsure|reject"
}

Use exactly this one field and one allowed enum value. Do not return Markdown,
a reasoning field, or any additional field. Think through the evidence
internally before returning the final JSON.

## Trust boundary

`username` is discovery context only. A shared searched username, its spelling
variants, or the current profile URL is never identity evidence.

The current extraction may still contain mistakes. Ignore placeholders
(`Not Found`, `Unknown`, `N/A`), zero or missing-data statements, account
telemetry, counts, ranks, scores, points, dates, status, platform labels,
navigation, system/API fields, breach artifacts, and feed/post content. These
facts cannot support either `strong_match` or `unsure`.

## The subject of the decision

`current_site.extraction` is the only subject. `anchors` and `strong_profile`
are reference evidence describing the person sought. They are never compared
to each other, and agreement between them is not evidence about the current
site.

Before classifying, list internally the usable identity facts that
`current_site.extraction` itself supplies, after discarding everything named
above. If that list is empty, return `reject`. A site that contributes no
usable fact cannot match, no matter how well `anchors` and `strong_profile`
agree with each other.

Every `strong_match` and every `unsure` must rest on at least one fact taken
from `current_site.extraction`.

## Classification

Return `strong_match` when explicit current-site owner facts clearly match the
supplied anchors or confirmed profile facts. Exact matches and clear semantic
equivalents both count, regardless of the field name. For example:

- Anchor role `ethical hacker`; current role `Penetration Tester` ->
  `strong_match`
- Anchor organization `Cloudflare`; current organization `Cloudflare` ->
  `strong_match`
- Anchor location `Portugal`; current location `Portugal` -> `strong_match`

Return `unsure` when the current facts are only partially, indirectly, or
ambiguously compatible with the supplied evidence. For example, anchor role
`ethical hacker` and current role `IT specialist` is related but too broad, so
return `unsure`.

Return `reject` when usable matching evidence is absent or when explicit
current-site facts conflict with the supplied evidence. A shared searched
username and lack of contradiction are not enough for either `strong_match` or
`unsure`.

Use all supplied anchor fields. Key names may be dynamic. Do not apply a hidden
fixed list of identity fields, and do not downgrade a clear semantic anchor
match merely because it is a role, organization, location, interest, skill, or
other profile fact.

## Personal names

A name anchor matches at the granularity it was supplied. If the anchor is a
name component and the current-site owner name contains that component, that
is an explicit match. Do not require the anchor to reproduce the full name,
and do not downgrade the match because the name is a common one.

Match on name components, not raw substrings. `erik` matches the given name in
`Erik T. Halvorsen`, but does not match `Frederik Baumann`.

- Anchor name `erik`; current owner name `Erik T. Halvorsen` -> `strong_match`
- Anchor name `erik`; current owner name `Erik` -> `strong_match`
- Anchor name `erik`; current owner name `Frederik Baumann` -> `reject`

Before answering, verify internally that the evidence supports the chosen enum.
