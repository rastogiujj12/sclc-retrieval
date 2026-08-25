# QASPER cross-section challenge analysis

This supplementary workflow evaluates only adjudicated questions whose complete
support genuinely requires evidence from multiple top-level sections.

The primary 180-question experiment is unchanged.

## Workflow

1. Run the complete QASPER audit.
2. Freeze the 53 strict held-out candidates.
3. Complete the blinded review before inspecting challenge retrieval results.
4. Finalise the review with `finalize_cross_section_challenge`.
5. Create a separate challenge configuration and selected-document manifest.
6. Run section-isolated, section-constrained, and global retrieval at 128, 256,
   and 512 tokens with Granite and Jina.
7. Run `analyse_cross_section_challenge` on the accepted query IDs.

The final analysis reports all six principal metrics and document-level paired
bootstrap comparisons. Granite is reported on all accepted questions and on the
common cross-model subset. Jina is reported on the common cross-model subset.

The challenge analysis is exploratory and must not be combined with the primary
held-out averages.
