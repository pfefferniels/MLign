"""Serialize an alignment into MEI's performance module.

Representation follows aligned-mei ("As Played By", research/02 §6) with the
three gaps closed per DESIGN §2: insertions and deletions are emitted, every
<when> carries a confidence <extData>, and wrong notes carry the played pitch.

Structure written INTO an existing MEI document (before </music>, or a
standalone fragment):

  <performance xml:id="mlign_perf">
    <recording xml:id="mlign_rec" source="#{source_id}">
      <!-- match -->
      <when absolute="1234ms" abstype="smil" data="#{score xml:id}"
            corresp="{perf id}">
        <extData type="velocity">64</extData>
        <extData type="duration">979ms</extData>
        <extData type="confidence">0.97</extData>
        <extData type="playedPitch">61</extData>   <!-- only if != score -->
      </when>
      <!-- insertion: no @data -->
      <when absolute="2000ms" abstype="smil" corresp="{perf id}" type="insertion">
        <extData type="pitch">72</extData> ...
      </when>
      <!-- deletion: no @absolute/@corresp -->
      <when data="#{score xml:id}" type="deletion"/>
    </recording>
  </performance>
"""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr


def _ext(t: str, v) -> str:
    return f'<extData type="{t}">{escape(str(v))}</extData>'


def alignment_to_performance_xml(
    triples: list[dict],
    perf_notes: dict[str, dict],
    source_id: str = "mlign_source",
) -> str:
    """triples: DESIGN §2 records. perf_notes: perf_id → {onset (s), duration
    (s), pitch, velocity}. Returns the <performance> element as text."""
    whens: list[str] = []
    for t in triples:
        label = t["label"]
        conf = t.get("confidence")
        if label == "match":
            p = perf_notes[t["perf_id"]]
            ext = [_ext("velocity", p["velocity"]), _ext("duration", f"{round(p['duration'] * 1000)}ms")]
            if conf is not None:
                ext.append(_ext("confidence", f"{conf:.3f}"))
            sub = t.get("sub")
            if sub:
                ext.append(_ext("playedPitch", sub["to"]))
            whens.append(
                f'<when absolute="{round(p["onset"] * 1000)}ms" abstype="smil" '
                f"data={quoteattr('#' + t['score_id'])} corresp={quoteattr(t['perf_id'])}>"
                + "".join(ext)
                + "</when>"
            )
        elif label == "insertion":
            p = perf_notes[t["perf_id"]]
            ext = [
                _ext("pitch", p["pitch"]),
                _ext("velocity", p["velocity"]),
                _ext("duration", f"{round(p['duration'] * 1000)}ms"),
            ]
            if conf is not None:
                ext.append(_ext("confidence", f"{conf:.3f}"))
            orn = t.get("ornament")
            if orn and orn.get("anchor_score_id"):
                ext.append(_ext("ornamentAnchor", "#" + orn["anchor_score_id"]))
            whens.append(
                f'<when absolute="{round(p["onset"] * 1000)}ms" abstype="smil" '
                f"corresp={quoteattr(t['perf_id'])} type=\"insertion\">"
                + "".join(ext)
                + "</when>"
            )
        elif label == "deletion":
            attrs = f"data={quoteattr('#' + t['score_id'])} type=\"deletion\""
            if conf is not None:
                whens.append(f"<when {attrs}>{_ext('confidence', f'{conf:.3f}')}</when>")
            else:
                whens.append(f"<when {attrs}/>")
    body = "\n      ".join(whens)
    return (
        '<performance xml:id="mlign_perf">\n'
        f'    <recording xml:id="mlign_rec" source={quoteattr("#" + source_id)}>\n'
        f"      {body}\n"
        "    </recording>\n"
        "  </performance>"
    )


def inject_into_mei(mei_text: str, performance_xml: str) -> str:
    """Insert before </music>'s closing tag (after the last </body> content).
    Falls back to appending before </mei>."""
    for anchor in ("</music>", "</mei>"):
        at = mei_text.rfind(anchor)
        if at != -1:
            return mei_text[:at] + "  " + performance_xml + "\n" + mei_text[at:]
    raise ValueError("no </music> or </mei> anchor found in MEI document")
